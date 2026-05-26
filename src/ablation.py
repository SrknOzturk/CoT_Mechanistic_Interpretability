"""
src/ablation.py
This module contains the logic for attention head ablation, 
including hook generation, random sampling strategies, and 
the end-to-end ablation execution pipelines.
"""

import torch
import random
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Optional, Any
from transformer_lens import HookedTransformer

# Import helpers from utils
from src.utils import (
    build_heads_by_example_id_from_curated,
    generate_with_optional_ablation,
    generate_cot_with_optional_ablation,
    make_direct_equation_prompt_from_row,
    make_nocot_prompt_from_row,
    make_cot_prompt_from_row,
    extract_last_number,
    numeric_equal,
    get_example_id_from_row,
    get_answer_from_row,
    get_type_from_row,
    generate_with_till_answer_optional_ablation
)

# ============================================================
# 1. Ablation Hook Logic
# ============================================================

def make_zero_ablation_hooks(heads: List[Tuple[int, int]]) -> List[Tuple[str, Callable]]:
    """
    Creates hooks that zero out the 'hook_z' activations for the specified heads.
    
    Args:
        heads: List of (layer, head) tuples to ablate.
    """
    if not heads:
        return []

    layer_dict = defaultdict(list)
    for layer, head in heads:
        layer_dict[int(layer)].append(int(head))

    hooks = []
    for layer, head_list in layer_dict.items():
        hook_name = f"blocks.{layer}.attn.hook_z"

        def hook_fn(z, hook, head_list=head_list):
            # Zero out the output of the specified heads at all positions
            for h in head_list:
                z[:, :, h, :] = 0.0
            return z

        hooks.append((hook_name, hook_fn))

    return hooks


def sample_random_heads_matched_layers_for_example(
    model: HookedTransformer, 
    selected_heads_list: List[Tuple[int, int]], 
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Randomly samples heads while maintaining the same layer distribution 
    as the input list (Layer-matched control).
    """
    random.seed(seed)
    n_heads = model.cfg.n_heads
    layer_counts = defaultdict(int)

    for layer, _ in selected_heads_list:
        layer_counts[int(layer)] += 1

    random_heads = []
    for layer, count in layer_counts.items():
        available_heads = list(range(n_heads))
        chosen = random.sample(available_heads, count)
        for h in chosen:
            random_heads.append((layer, h))

    return random_heads


def sample_random_heads_same_count_for_example(
    model: HookedTransformer, 
    num_heads: int, 
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Randomly samples heads from the entire pool of model heads.
    """
    random.seed(seed)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    if num_heads > len(all_heads):
        num_heads = len(all_heads)

    return random.sample(all_heads, num_heads)


# ============================================================
# 2. Experiment Pipelines
# ============================================================

def run_direct_equation_ablation_using_curated_heads(
    sampled_df: pd.DataFrame,
    model: HookedTransformer,
    curated_normal_results: List[Dict],
    selected_heads_key: str = "selected_heads",
    max_examples: Optional[int] = None,
    max_new_tokens: int = 2048,
    random_seed: int = 42,
    print_tokens: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Runs Direct Equation ablation experiments comparing curated heads vs. random heads.
    """
    from tqdm import tqdm
    heads_by_example_id = build_heads_by_example_id_from_curated(curated_normal_results, selected_heads_key)
    results = []

    total = len(sampled_df) if max_examples is None else min(len(sampled_df), max_examples)
    
    for idx, row in tqdm(sampled_df.iterrows(), total=total, desc="Running Direct Equation Ablation"):
        if max_examples is not None and idx >= max_examples:
            break

        equation_prompt = make_direct_equation_prompt_from_row(row)
        if equation_prompt is None: continue

        example_id = get_example_id_from_row(row, idx)
        true_ans = get_answer_from_row(row)
        q_type = get_type_from_row(row)

        selected_heads_list = heads_by_example_id.get(example_id, [])
        num_heads = len(selected_heads_list)

        # 1. Normal Generation
        normal_text = generate_with_till_answer_optional_ablation(model, equation_prompt, ablated_heads=None)
        
        # 2. Selected-head Ablation
        ablation_text = generate_with_till_answer_optional_ablation(
            model, equation_prompt, ablated_heads=selected_heads_list
        ) if num_heads > 0 else None

        # 3. Random-head Ablation
        random_heads_list = sample_random_heads_matched_layers_for_example(model, selected_heads_list, seed=random_seed + idx) if num_heads > 0 else None
        random_text = generate_with_till_answer_optional_ablation(
            model, equation_prompt, ablated_heads=random_heads_list
        ) if num_heads > 0 else None

        results.append({
            "example_id": example_id,
            "Type": q_type,
            "normal_correct": numeric_equal(extract_last_number(normal_text), true_ans),
            "ablation_correct": numeric_equal(extract_last_number(ablation_text), true_ans) if ablation_text else False,
            "random_correct": numeric_equal(extract_last_number(random_text), true_ans) if random_text else False,
        })

    return pd.DataFrame(results)


def run_nocot_ablation_using_curated_heads(
    sampled_df: pd.DataFrame,
    model: HookedTransformer,
    curated_normal_results: List[Dict],
    selected_heads_key: str = "selected_heads",
    max_examples: Optional[int] = None,
    max_new_tokens: int = 2048,
    random_seed: int = 42,
    print_tokens: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Runs No-CoT ablation experiments.
    """
    from tqdm import tqdm
    heads_by_example_id = build_heads_by_example_id_from_curated(curated_normal_results, selected_heads_key)
    results = []

    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Running No-CoT Ablation"):
        if max_examples is not None and idx >= max_examples: break

        example_id = get_example_id_from_row(row, idx)
        true_ans = get_answer_from_row(row)
        nocot_prompt = make_nocot_prompt_from_row(row)
        selected_heads_list = heads_by_example_id.get(example_id, [])
        num_heads = len(selected_heads_list)

        normal_text = generate_with_optional_ablation(model, nocot_prompt, ablated_heads=None)
        ablation_text = generate_with_optional_ablation(model, nocot_prompt, ablated_heads=selected_heads_list) if num_heads > 0 else None
        random_heads_list = sample_random_heads_same_count_for_example(model, num_heads, seed=random_seed + idx) if num_heads > 0 else None
        random_text = generate_with_optional_ablation(model, nocot_prompt, ablated_heads=random_heads_list) if num_heads > 0 else None

        results.append({
            "example_id": example_id,
            "normal_correct": numeric_equal(extract_last_number(normal_text), true_ans),
            "ablation_correct": numeric_equal(extract_last_number(ablation_text), true_ans) if ablation_text else False,
            "random_correct": numeric_equal(extract_last_number(random_text), true_ans) if random_text else False,
        })
    return pd.DataFrame(results)


def run_cot_ablation_using_curated_heads(
    sampled_df: pd.DataFrame,
    model: HookedTransformer,
    curated_normal_results: List[Dict],
    selected_heads_key: str = "selected_heads",
    max_examples: Optional[int] = None,
    max_cot_reasoning_tokens: int = 2048,
    random_seed: int = 42,
    print_tokens: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Runs CoT ablation experiments.
    """
    from tqdm import tqdm
    heads_by_example_id = build_heads_by_example_id_from_curated(curated_normal_results, selected_heads_key)
    results = []

    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Running CoT Ablation"):
        if max_examples is not None and idx >= max_examples: break

        example_id = get_example_id_from_row(row, idx)
        true_ans = get_answer_from_row(row)
        cot_prompt = make_cot_prompt_from_row(row)
        selected_heads_list = heads_by_example_id.get(example_id, [])
        num_heads = len(selected_heads_list)

        normal_text = generate_cot_with_optional_ablation(model, cot_prompt, ablated_heads=None)
        ablation_text = generate_cot_with_optional_ablation(model, cot_prompt, ablated_heads=selected_heads_list) if num_heads > 0 else None
        random_heads_list = sample_random_heads_same_count_for_example(model, num_heads, seed=random_seed + idx) if num_heads > 0 else None
        random_text = generate_cot_with_optional_ablation(model, cot_prompt, ablated_heads=random_heads_list) if num_heads > 0 else None

        results.append({
            "example_id": example_id,
            "normal_correct": numeric_equal(extract_last_number(normal_text), true_ans),
            "ablation_correct": numeric_equal(extract_last_number(ablation_text), true_ans) if ablation_text else False,
            "random_correct": numeric_equal(extract_last_number(random_text), true_ans) if random_text else False,
        })
    return pd.DataFrame(results)