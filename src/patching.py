"""
src/patching.py
This module contains the core activation patching
These functions intervene in the model's computation graph using TransformerLens hooks.
"""

import torch
from collections import defaultdict
from typing import Callable, List, Tuple, Any
from transformer_lens import HookedTransformer


def patch_attn_head_out_last_pos(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    metric_fn: Callable,
    normalize: bool = False
) -> torch.Tensor:
    """
    Activation patching over attention-head outputs (hook_z) at the **last token position**.

    Args:
        model (HookedTransformer): TransformerLens-style model that supports hooks.
        corrupted_tokens (torch.Tensor): Tokenized corrupted input, shape [batch, seq].
        clean_cache: Cache from a clean run, must contain entries like clean_cache[("z", layer)] 
                     with shape [batch, seq, n_heads, d_head].
        metric_fn (Callable): Function that takes logits and returns a scalar score.
        normalize (bool, optional): If True, the logits are Z-score normalized (mean=0, std=1)
                                    before being passed to the metric function. This helps in
                                    comparing results across different scales. Defaults to False.

    Returns:
        torch.Tensor: A [n_layers, n_heads] tensor of scores — higher means that 
                      the head's clean activation (at the last position) helps more.
    """
    corrupted_seq_len = corrupted_tokens.shape[1]
    corrupted_last_pos = corrupted_seq_len - 1

    # Get clean cache sequence length (clean_tokens might be a different length)
    # Use the first layer's z cache to determine the clean sequence length
    first_layer_z = clean_cache["z", 0]
    clean_seq_len = first_layer_z.shape[1]
    clean_last_pos = clean_seq_len - 1

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Allocate the score matrix
    scores = torch.zeros(n_layers, n_heads, device=corrupted_tokens.device)

    # Choose the correct hook name for the attention head OUTPUT ("z") in TransformerLens.
    hook_name_template = "blocks.{layer}.attn.hook_z"

    # Sweep over all layers and heads, patching one head at a time
    for layer in range(n_layers):
        for head in range(n_heads):
            # Capture layer and head in closure to avoid late binding issues
            layer_idx = layer
            head_idx = head

            def hook_fn(value, hook):
                """
                Hook is called at the attention 'z' output:
                    value shape: [batch, seq, n_heads, d_head]

                We copy `value`, then replace ONLY the vector at:
                    - sequence position: corrupted_last_pos
                    - head index       : `head_idx`
                with the clean activation from `clean_cache` at:
                    - sequence position: clean_last_pos
                    - head index       : `head_idx`
                """
                # Defensive copy so we don't mutate the original tensor in-place
                value = value.clone()

                # Retrieve the clean 'z' activations for this layer
                clean_value = clean_cache["z", layer_idx]

                # Patch: inject the clean activation into the corrupted run at the last position
                value[:, corrupted_last_pos, head_idx, :] = clean_value[:, clean_last_pos, head_idx, :]
                return value

            # Build the concrete hook name for this layer
            hook_name = hook_name_template.format(layer=layer)

            # Run the model once with this single hook active
            with torch.no_grad():
                logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits"
                )

            # Convert logits to a scalar via the provided metric
            if normalize:
                logits = (logits - logits.mean()) / logits.std()
            scores[layer, head] = metric_fn(logits)

    return scores


def patch_attn_head_out_last_pos_with_zero_ablation(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    metric_fn: Callable,
    normalize: bool = False,
    ablate_layer: int = 13,
    ablate_head: int = 12,
) -> torch.Tensor:
    """
    Activation patching over attention-head outputs (hook_z) at the last token position,
    while ALSO zero-ablating a fixed (ablate_layer, ablate_head) on the corrupted run.

    This is used to measure the effect of patching into a model where a specific head 
    has been zero-ablated. During patching:
      - The targeted ablate_head is zeroed out at the last position across all runs.
      - If the head currently being patched is the same as the ablated head, the patch 
        function overwrites the zeros with the clean value (patching overrides ablation).
    """
    corrupted_seq_len = corrupted_tokens.shape[1]
    corrupted_last_pos = corrupted_seq_len - 1

    first_layer_z = clean_cache["z", 0]
    clean_seq_len = first_layer_z.shape[1]
    clean_last_pos = clean_seq_len - 1

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    scores = torch.zeros(n_layers, n_heads, device=corrupted_tokens.device)

    hook_name_template = "blocks.{layer}.attn.hook_z"

    for layer in range(n_layers):
        for head in range(n_heads):
            layer_idx = layer
            head_idx = head

            def ablate_fn(value, hook):
                v = value.clone()
                if 0 <= ablate_head < v.shape[2] and 0 <= ablate_layer < model.cfg.n_layers:
                    last_pos = corrupted_last_pos
                    # Zero out the ablated head only at the last position
                    if hook.name == hook_name_template.format(layer=ablate_layer):
                        v[:, last_pos, ablate_head, :] = 0.0
                return v

            def patch_fn(value, hook):
                v = value.clone()
                clean_value = clean_cache["z", layer_idx]
                v[:, corrupted_last_pos, head_idx, :] = clean_value[:, clean_last_pos, head_idx, :]
                return v

            hooks = []
            # Add the ablation hook only at the corresponding layer
            hook_name_ablate = hook_name_template.format(layer=ablate_layer)
            hooks.append((hook_name_ablate, ablate_fn))

            hook_name_patch = hook_name_template.format(layer=layer_idx)
            hooks.append((hook_name_patch, patch_fn))

            with torch.no_grad():
                logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=hooks,
                    return_type="logits",
                )

            if normalize:
                logits = (logits - logits.mean()) / logits.std()
            scores[layer, head] = metric_fn(logits)

    return scores


def get_patched_logits_for_head(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    layer_idx: int,
    head_idx: int
) -> torch.Tensor:
    """
    Applies patching for a specific (layer, head) pair and returns the resulting logits.

    Args:
        model: HookedTransformer model
        corrupted_tokens: Tokenized corrupted input
        clean_cache: Cache retrieved from the clean run
        layer_idx: Layer index
        head_idx: Head index

    Returns:
        torch.Tensor: Patched logits [batch, seq, vocab_size]
    """
    corrupted_seq_len = corrupted_tokens.shape[1]
    corrupted_last_pos = corrupted_seq_len - 1

    first_layer_z = clean_cache["z", 0]
    clean_seq_len = first_layer_z.shape[1]
    clean_last_pos = clean_seq_len - 1

    hook_name_template = "blocks.{layer}.attn.hook_z"

    def hook_fn(value, hook):
        value = value.clone()
        clean_value = clean_cache["z", layer_idx]
        value[:, corrupted_last_pos, head_idx, :] = clean_value[:, clean_last_pos, head_idx, :]
        return value

    hook_name = hook_name_template.format(layer=layer_idx)

    with torch.no_grad():
        logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits"
        )

    return logits


def get_patched_logits_for_heads(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    head_specs: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Simultaneously patches multiple (layer, head) pairs and returns the resulting logits.
    
    Args:
        model: HookedTransformer model
        corrupted_tokens: Tokenized corrupted input
        clean_cache: Cache retrieved from the clean run
        head_specs: List of tuples [(layer_idx, head_idx), ...], e.g., [(16,6), (16,7)]

    Returns:
        torch.Tensor: Patched logits [batch, seq, vocab_size]
    """
    corrupted_seq_len = corrupted_tokens.shape[1]
    corrupted_last_pos = corrupted_seq_len - 1

    first_layer_z = clean_cache["z", 0]
    clean_seq_len = first_layer_z.shape[1]
    clean_last_pos = clean_seq_len - 1

    # Group by layer: {layer: [head1, head2, ...]}
    layer_to_heads = defaultdict(list)
    for (layer_idx, head_idx) in head_specs:
        layer_to_heads[layer_idx].append(head_idx)

    hook_name_template = "blocks.{layer}.attn.hook_z"
    fwd_hooks = []

    for layer_idx, head_indices in layer_to_heads.items():
        def make_hook(lyr, heads):
            def hook_fn(value, hook):
                value = value.clone()
                clean_value = clean_cache["z", lyr]
                for h in heads:
                    value[:, corrupted_last_pos, h, :] = clean_value[:, clean_last_pos, h, :]
                return value
            return hook_fn

        hook_name = hook_name_template.format(layer=layer_idx)
        fwd_hooks.append((hook_name, make_hook(layer_idx, head_indices)))

    with torch.no_grad():
        logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=fwd_hooks,
            return_type="logits"
        )

    return logits


def patch_attn_head_out_last_pos_random(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    metric_fn: Callable,
    normalize: bool = False
) -> torch.Tensor:
    """
    Activation patching with **random** tensors at the last position (baseline).
    For each head, replaces the activation at the last position with a randomly
    generated tensor of the same shape. Used to compare random vs normal (clean)
    patching at the start of an experiment.

    Returns:
        torch.Tensor: A [n_layers, n_heads] tensor of scores.
    """
    corrupted_seq_len = corrupted_tokens.shape[1]
    corrupted_last_pos = corrupted_seq_len - 1

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    batch = corrupted_tokens.shape[0]
    device = corrupted_tokens.device
    
    # Activations (z) are always float; never use token dtype (Long)
    dtype = torch.float32
    if clean_cache is not None and ("z", 0) in clean_cache:
        dtype = clean_cache["z", 0].dtype

    scores = torch.zeros(n_layers, n_heads, device=device)

    hook_name_template = "blocks.{layer}.attn.hook_z"

    for layer in range(n_layers):
        for head in range(n_heads):
            layer_idx = layer
            head_idx = head

            def hook_fn(value, hook):
                value = value.clone()
                # Random tensor: same shape [batch, d_head]
                random_vec = torch.randn(batch, d_head, device=device, dtype=dtype)
                value[:, corrupted_last_pos, head_idx, :] = random_vec
                return value

            hook_name = hook_name_template.format(layer=layer)

            with torch.no_grad():
                logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits"
                )

            if normalize:
                logits = (logits - logits.mean()) / logits.std()
            scores[layer, head] = metric_fn(logits)

    return scores