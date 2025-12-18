import torch
from tqdm import tqdm
from typing import List, Union, Callable, Optional
from transformer_lens import HookedTransformer


def patch_attn_head_out_last_pos(
    model: "HookedTransformer",
    corrupted_tokens: "torch.Tensor",
    clean_cache,
    metric_fn,
    normalize: bool = False
):
    """
    Activation patching over attention-head outputs (hook_z) at the **last token position**.

    Args:
        model (HookedTransformer): TransformerLens-style model that supports hooks.
        corrupted_tokens (torch.Tensor): Tokenized corrupted input, shape [batch, seq].
        clean_cache: Cache from a clean run, must contain entries like clean_cache[("z", layer)] with shape[batch, seq, n_heads, d_head]
        metric_fn (Callable): Function that takes logits and returns a scalar score
        normalize (bool, optional): If True, the logits are Z-score normalized (mean=0, std=1)
                                     before being passed to the metric function. This helps in
                                     comparing results across different scales. Defaults to False.

    Returns:
        torch.Tensor: A [n_layers, n_heads] tensor of scores — higher means that
                      head's clean activation (at the last position) helps more.
    """

    seq_len = corrupted_tokens.shape[1]
    last_pos = seq_len - 1

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Allocate the score matrix
    scores = torch.zeros(n_layers, n_heads, device=corrupted_tokens.device)

    # Choose the correct hook name for the attention head OUTPUT ("z") in TransformerLens.
    hook_name_template = "blocks.{layer}.attn.hook_z"

    # Sweep over all layers and heads, patching one head at a time
    for layer in range(n_layers):
        for head in range(n_heads):

            def hook_fn(value, hook, layer=layer, head=head):
                """
                Hook is called at the attention 'z' output:
                    value shape: [batch, seq, n_heads, d_head]

                We copy `value`, then replace ONLY the vector at:
                    - sequence position: last_pos
                    - head index       : `head`
                with the clean activation from `clean_cache`.
                """
                # Defensive copy so we don't mutate the original tensor in-place
                value = value.clone()

                # Retrieve the clean 'z' activations for this layer.
                clean_value = clean_cache["z", layer]  # or clean_cache[("z", layer)]

                # Patch only the last position and only this specific head across the entire batch
                value[:, last_pos, head, :] = clean_value[:, last_pos, head, :]
                return value

            # Build the concrete hook name for this layer
            hook_name = hook_name_template.format(layer=layer)

            # Run the model once with this single hook active.
            with torch.no_grad():
                logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits"
                )

            # Convert logits to a scalar via the provided metric
            if normalize == True:
              logits = (logits - logits.mean()) / logits.std()
            scores[layer, head] = metric_fn(logits)

    return scores


def patching_pipeline(
    model,
    clean_prompts: Union[str, List[str]],
    corrupted_prompts: Union[str, List[str]],
    patching_function: Callable = None,  # Function that performs the actual patching experiment
    metric: Callable = None,              # Metric factory: takes clean_logits -> returns fn(patched_logits) -> scalar
    padding: bool = False,                 # Whether to pad the inputs to the same length
    normalize: bool = False
):
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


    ctx = getattr(model.cfg, "n_ctx", 1024) # Context window (default fallback = 1024)

    if metric is None:
        raise ValueError(
            "The 'metric' parameter cannot be None. Please provide a metric factory "
            "such as 'make_kl_metric' from src.patching_evaluation_metrics."
        )

    if patching_function is None:
        raise ValueError(
            "The 'patching_function' parameter cannot be None. Please provide a function "
            "like 'patch_attn_head_by_position' or 'patch_attn_head_out_last_pos'."
        )


    patching_results = []

    with torch.no_grad():
        for clean_prompt, corrupted_prompt in tqdm(
            zip(clean_prompts, corrupted_prompts),
            total=len(clean_prompts),
            desc="Running patching pipeline",
            ncols=100,
            colour="cyan"
        ):

            # Tokenize both inputs and truncate to last `ctx` tokens
            clean_tokens     = model.to_tokens(clean_prompt)
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
                  #print()
                  #print(f"Tokens aligned → clean_len={len(clean_tokens[0])}, corrupted_len={len(corrupted_tokens[0])}")
                  #print(model.to_string(clean_tokens))
                  #print(model.to_string(corrupted_tokens))


            corrupted_tokens = corrupted_tokens[:, -ctx:].to(device)
            clean_tokens = clean_tokens[:,-ctx:].to(device)

            # Run clean pass and capture activations
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)

            # Prepare metric bound to the clean logits
            metric_function = metric(clean_logits)  # metric_function(patched_logits) -> scalar

            # Run the provided patching function
            patching_result = patching_function(
                model=model,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                metric_fn=metric_function,
                normalize = normalize
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





















