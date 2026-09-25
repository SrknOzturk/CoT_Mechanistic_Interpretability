"""
src/patching.py
This module contains the core activation patching
These functions intervene in the model's computation graph using TransformerLens hooks.
"""

import torch
from collections import defaultdict
from typing import Callable, List, Tuple, Any
from transformer_lens import HookedTransformer

from src.utils import logits_from_final_residual


def _capture_resid_pre(model: HookedTransformer, corrupted_tokens: torch.Tensor) -> dict:
    """
    The residual entering every block, from one unpatched run.

    Patching a component of layer L leaves blocks 0..L-1 untouched, so their
    output is the same for every unit of every layer in a sweep. Each sweep
    pass can therefore resume from resid_pre[L] and run n_layers-L blocks
    instead of all n_layers, which halves the sweep's work on average.

    Safe because the sweeps are batch-1 and unpadded (patching_pipeline's
    `padding` is off, and nothing enables it), so there is no attention mask
    to reconstruct from tokens that are no longer passed.
    """
    n_layers = model.cfg.n_layers
    resid_pre = {}

    def make_capture(idx):
        def capture(value, hook):
            resid_pre[idx] = value.clone()
            return value
        return capture

    with torch.no_grad():
        model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(f"blocks.{idx}.hook_resid_pre", make_capture(idx))
                       for idx in range(n_layers)],
            return_type=None,
            stop_at_layer=n_layers,
        )
    return resid_pre


def patch_mlp_out_last_pos(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache,
    metric_fn: Callable,
    normalize: bool = False
) -> torch.Tensor:
    """
    The MLP counterpart of patch_attn_head_out_last_pos: patches one layer's
    whole MLP output (hook_mlp_out) at the last position per forward pass.

    Returns [n_layers, 1] (or a dict of them), one unit per layer with head
    index 0, so the head-selection, joint-patching and record code downstream
    runs unchanged.
    """
    corrupted_last_pos = corrupted_tokens.shape[1] - 1
    clean_last_pos = clean_cache["mlp_out", 0].shape[1] - 1
    n_layers = model.cfg.n_layers

    single = not isinstance(metric_fn, dict)
    metric_fns = {"_": metric_fn} if single else metric_fn
    scores = {name: torch.zeros(n_layers, 1, device=corrupted_tokens.device)
              for name in metric_fns}

    resid_pre = _capture_resid_pre(model, corrupted_tokens)

    for layer in range(n_layers):
        clean_value = clean_cache["mlp_out", layer][:, clean_last_pos, :]

        def hook_fn(value, hook, clean_value=clean_value):
            value = value.clone()
            value[:, corrupted_last_pos, :] = clean_value
            return value

        with torch.no_grad():
            residual = model.run_with_hooks(
                resid_pre[layer],
                fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", hook_fn)],
                return_type=None,
                start_at_layer=layer,
                stop_at_layer=n_layers,
            )
            logits = logits_from_final_residual(model, residual)
            del residual

        if normalize:
            logits = (logits - logits.mean()) / logits.std()
        for name, fn in metric_fns.items():
            scores[name][layer, 0] = fn(logits)
        del logits

    return scores["_"] if single else scores


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
        metric_fn (Callable | dict): Either one function mapping logits to a scalar,
                     or a {name: function} dict. A dict scores every metric from the
                     same forward pass and returns one heatmap per name, which is how
                     margin and JSD are obtained without sweeping the trace twice.
        normalize (bool, optional): If True, the logits are Z-score normalized (mean=0, std=1)
                                    before being passed to the metric function. This helps in
                                    comparing results across different scales. Defaults to False.

    Returns:
        torch.Tensor | dict: A [n_layers, n_heads] tensor of scores, or a dict of
                      them when metric_fn is a dict. Higher means the head's clean
                      activation (at the last position) helps more -- except for
                      divergence metrics such as JSD, where lower is better.
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

    # one score matrix per metric; a bare callable is treated as a single unnamed one
    single = not isinstance(metric_fn, dict)
    metric_fns = {"_": metric_fn} if single else metric_fn
    scores = {name: torch.zeros(n_layers, n_heads, device=corrupted_tokens.device)
              for name in metric_fns}

    # Choose the correct hook name for the attention head OUTPUT ("z") in TransformerLens.
    hook_name_template = "blocks.{layer}.attn.hook_z"

    resid_pre = _capture_resid_pre(model, corrupted_tokens)

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

            # Resume from this layer's cached prefix and stop before the unembed.
            # Every metric reads logits[0, -1, :] only, so unembedding the whole
            # sequence into a 152k-token vocabulary -- once per head, so
            # n_layers*n_heads times per token step, on a sequence that grows
            # with the trace -- was the dominant cost here and made long traces
            # run out of memory outright.
            with torch.no_grad():
                residual = model.run_with_hooks(
                    resid_pre[layer],
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type=None,
                    start_at_layer=layer,
                    stop_at_layer=n_layers,
                )
                logits = logits_from_final_residual(model, residual)
                del residual

            # Score every metric from this one forward pass
            if normalize:
                logits = (logits - logits.mean()) / logits.std()
            for name, fn in metric_fns.items():
                scores[name][layer, head] = fn(logits)
            del logits

    return scores["_"] if single else scores


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