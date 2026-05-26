"""
src/patching_pipelines.py
This module orchestrates the activation patching process across datasets.
It handles tokenization, sequence padding, cache generation, and metric aggregation
over pairs of clean and corrupted prompts.
"""

import torch
from typing import Union, List, Callable
from transformer_lens import HookedTransformer


def patching_pipeline(
    model: HookedTransformer,
    clean_prompts: Union[str, List[str]],
    corrupted_prompts: Union[str, List[str]],
    patching_function: Callable = None,
    metric: Callable = None,
    padding: bool = False,
    normalize: bool = False,
    clean_reference_logits: torch.Tensor = None,
) -> torch.Tensor:
    """
    Runs an activation-patching sweep over attention heads, averaging scores across prompt pairs.
    Adds optional automatic left-padding to equalize token sequence lengths per pair.

    Parameters
    ----------
    model : HookedTransformer
        TransformerLens-compatible model.
    clean_prompts : str | list[str]
        Clean input prompt(s).
    corrupted_prompts : str | list[str]
        Corrupted input prompt(s), same length as clean_prompts.
    patching_function : Callable, optional
        Signature: patching_function(model, corrupted_tokens, clean_cache, metric_fn) -> torch.Tensor [L, H]
        Defaults to `patch_attn_head_out_last_pos` if not provided.
    metric : Callable, optional
        Metric factory: metric(clean_logits) -> (patched_logits -> scalar). Default = KL-based metric.
    padding : bool, default False
        If True, shorter prompt in each pair is left-padded to match the longer one.
    normalize : bool, default False
        If True, indicates that the internal patching function should normalize logits
        before calculating the final metric.
    clean_reference_logits: torch.Tensor, optional
        Fixed reference logits to use for the metric instead of the logits generated in the current run.

    Returns
    -------
    torch.Tensor
        [n_layers, n_heads] average score across all prompt pairs.
    """

    # --- Preparation
    model.reset_hooks()
    model.eval()

    if isinstance(clean_prompts, str):
        clean_prompts = [clean_prompts]
    if isinstance(corrupted_prompts, str):
        corrupted_prompts = [corrupted_prompts]

    assert len(clean_prompts) == len(corrupted_prompts), \
        "clean_prompts and corrupted_prompts must have the same length."

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    ctx = getattr(model.cfg, "n_ctx", 1024)  # Context window (default fallback = 1024)

    if metric is None:
        raise ValueError(
            "The 'metric' parameter cannot be None. Please provide a metric factory "
            "such as 'make_kl_metric' from src.metrics."
        )

    if patching_function is None:
        raise ValueError(
            "The 'patching_function' parameter cannot be None. Please provide a function "
            "like 'patch_attn_head_out_last_pos'."
        )

    patching_results = []

    with torch.no_grad():
        for clean_prompt, corrupted_prompt in zip(clean_prompts, corrupted_prompts):

            # Tokenize both inputs and truncate to last `ctx` tokens
            clean_tokens = model.to_tokens(clean_prompt)
            corrupted_tokens = model.to_tokens(corrupted_prompt)

            if padding:
                clean_len = len(clean_tokens[0])
                corrupted_len = len(corrupted_tokens[0])

                if clean_len != corrupted_len:
                    max_len = max(clean_len, corrupted_len)
                    pad_token = model.tokenizer.bos_token_id if hasattr(model.tokenizer, "bos_token_id") else model.tokenizer.pad_token_id

                    if clean_len < max_len:
                        pad_count = max_len - clean_len
                        pad_tensor = torch.full((1, pad_count), pad_token, dtype=clean_tokens.dtype, device=clean_tokens.device)
                        clean_tokens = torch.cat([pad_tensor, clean_tokens], dim=1)

                    if corrupted_len < max_len:
                        pad_count = max_len - corrupted_len
                        pad_tensor = torch.full((1, pad_count), pad_token, dtype=corrupted_tokens.dtype, device=corrupted_tokens.device)
                        corrupted_tokens = torch.cat([pad_tensor, corrupted_tokens], dim=1)

            corrupted_tokens = corrupted_tokens[:, -ctx:].to(device)
            clean_tokens = clean_tokens[:, -ctx:].to(device)

            # IMPORTANT: Run the clean pass from scratch and capture activations
            # In each iteration, we run the clean prompt from scratch to obtain the clean_cache
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)

            # Determine which clean logits to use for the metric
            if clean_reference_logits is not None:
                # Use fixed reference (e.g., the last token of the full answer generated initially)
                # clean_reference_logits is shaped [vocab_size]; we need to unsqueeze it to [1, 1, vocab_size]
                clean_reference_logits_expanded = clean_reference_logits.unsqueeze(0).unsqueeze(0)
                metric_function = metric(clean_reference_logits_expanded)
            else:
                # Default behavior: use the current clean logits
                metric_function = metric(clean_logits)  # metric_function(patched_logits) -> scalar

            # Run the provided patching function
            patching_result = patching_function(
                model=model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                metric_fn=metric_function,
                normalize=normalize
            )

            # Move result to CPU and store
            if torch.is_tensor(patching_result):
                patching_result = patching_result.detach().cpu()
            patching_results.append(patching_result)

            # Free memory between iterations
            del clean_cache, clean_logits, clean_tokens, corrupted_tokens
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    # Average over all pairs -> final importance map
    avg_patching_result = torch.stack(patching_results, dim=0).mean(dim=0)
    return avg_patching_result


def patching_pipeline_with_relative_metric(
    model: HookedTransformer,
    clean_prompts: Union[str, List[str]],
    corrupted_prompts: Union[str, List[str]],
    patching_function: Callable = None,
    metric: Callable = None,
    padding: bool = False,
    normalize: bool = False
) -> torch.Tensor:
    """
    Runs an activation-patching sweep over attention heads, averaging scores across prompt pairs.
    Designed specifically for relative metrics that require both clean and corrupted reference logits.
    Adds optional automatic left-padding to equalize token sequence lengths per pair.

    Parameters
    ----------
    ... [Same standard parameters as patching_pipeline]
    metric : Callable, optional
        Metric factory: metric(clean_logits, corrupted_logits) -> (patched_logits -> scalar).
    """

    # --- Preparation
    model.reset_hooks()
    model.eval()

    if isinstance(clean_prompts, str):
        clean_prompts = [clean_prompts]
    if isinstance(corrupted_prompts, str):
        corrupted_prompts = [corrupted_prompts]

    assert len(clean_prompts) == len(corrupted_prompts), \
        "clean_prompts and corrupted_prompts must have the same length."

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    ctx = getattr(model.cfg, "n_ctx", 1024)  # Context window (default fallback = 1024)

    if metric is None:
        raise ValueError(
            "The 'metric' parameter cannot be None. Please provide a metric factory."
        )

    if patching_function is None:
        raise ValueError(
            "The 'patching_function' parameter cannot be None. Please provide a function."
        )

    patching_results = []

    with torch.no_grad():
        for clean_prompt, corrupted_prompt in zip(clean_prompts, corrupted_prompts):

            # Tokenize both inputs and truncate to last `ctx` tokens
            clean_tokens = model.to_tokens(clean_prompt)
            corrupted_tokens = model.to_tokens(corrupted_prompt)

            if padding:
                clean_len = len(clean_tokens[0])
                corrupted_len = len(corrupted_tokens[0])

                if clean_len != corrupted_len:
                    max_len = max(clean_len, corrupted_len)
                    pad_token = model.tokenizer.bos_token_id if hasattr(model.tokenizer, "bos_token_id") else model.tokenizer.pad_token_id

                    if clean_len < max_len:
                        pad_count = max_len - clean_len
                        pad_tensor = torch.full((1, pad_count), pad_token, dtype=clean_tokens.dtype, device=clean_tokens.device)
                        clean_tokens = torch.cat([pad_tensor, clean_tokens], dim=1)

                    if corrupted_len < max_len:
                        pad_count = max_len - corrupted_len
                        pad_tensor = torch.full((1, pad_count), pad_token, dtype=corrupted_tokens.dtype, device=corrupted_tokens.device)
                        corrupted_tokens = torch.cat([pad_tensor, corrupted_tokens], dim=1)

            corrupted_tokens = corrupted_tokens[:, -ctx:].to(device)
            clean_tokens = clean_tokens[:, -ctx:].to(device)

            # Run clean & corrupted passes and capture activations
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

            # Prepare metric bound to BOTH the clean and corrupted logits
            metric_function = metric(clean_logits, corrupted_logits)

            # Run the provided patching function
            patching_result = patching_function(
                model=model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                metric_fn=metric_function,
                normalize=normalize
            )

            # Move result to CPU and store
            if torch.is_tensor(patching_result):
                patching_result = patching_result.detach().cpu()
            patching_results.append(patching_result)

            # Free memory between iterations
            del clean_cache, clean_logits, clean_tokens, corrupted_tokens, corrupted_logits, corrupted_cache
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    # Average over all pairs -> final importance map
    avg_patching_result = torch.stack(patching_results, dim=0).mean(dim=0)
    return avg_patching_result