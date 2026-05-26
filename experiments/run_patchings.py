"""
experiments/run_patching.py

Executes the main Sequential Multi-Head Patching experiments:
1. Normal Patching (Margin)
2. Cross Patching (Margin)
3. Normal Patching (JSD)
4. Cross Patching (JSD)
5. Random Patching Control (Margin)
6. Random Patching Control (JSD)
"""

import sys
import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
import nltk
from tqdm.auto import tqdm

# Add the root project directory to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import (
    load_model,
    generate_full_answer_and_get_logits,
    generate_till_answer,
    _merge_top_heads_for_pos_margin_ratio,
    _merge_top_heads_for_pos_jsd
)
from src.metrics import (
    make_margin_recovery_ratio_metric,
    make_jsd_metric,
    margin_recovery_ratio
)
from src.patching import patch_attn_head_out_last_pos
from src.patching_pipelines import patching_pipeline


# =============================================================================
# 1. Normal Patching Functions
# =============================================================================

def multi_head_patching_with_margin_difference(
    df,
    model,
    id_column="ID", 
    ctx=1024,
    margin_ratio_heads_per_pos=3,
    max_generation_steps=500,
    output_json_path="multi_head_patching_with_margin_results.json"
):
    """
    Performs the multi-head patching experiment for each example in the given DataFrame.
    Saves full hm_ld matrices, probabilities, token IDs, and all main metrics to a 
    hierarchically structured JSON file.
    """
    metric_factory = make_margin_recovery_ratio_metric
    device = next(model.parameters()).device
    json_export_data = [] 

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Examples (Margin)"):
        example_id = row[id_column]
        cot_prompt_for_reference = row["PromptWithCot"]
        cot_prompt_for_patching = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]
        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(
            model, cot_prompt_for_reference
        )
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_probs = F.softmax(no_cot_logits, dim=-1)
        no_cot_t_true_prob = no_cot_probs[t_true].item()

        active_metric_factory = metric_factory(no_cot_logits)

        prompt_ld = cot_prompt_for_patching
        best_head_by_pos_normal_ld = {}
        pos_hm_matrices = defaultdict(list)
        step = 0
        token_level_results = []

        while not prompt_ld.endswith("The answer is ") and step < max_generation_steps:
            hm_ld = patching_pipeline(
                model,
                prompt_ld,
                no_cot_prompt_last_token,
                metric=active_metric_factory,
                patching_function=patch_attn_head_out_last_pos,
                clean_reference_logits=clean_reference_logits,
            )

            prompt_tokens_ld = model.to_tokens(prompt_ld)
            last_token_id = prompt_tokens_ld[0, -1].item()
            last_token_str_ld = model.to_string(prompt_tokens_ld[:, -1:])[0].strip()

            pos_tagged_ld = nltk.pos_tag([last_token_str_ld] if last_token_str_ld else ["."])
            pos_label_ld = pos_tagged_ld[0][1] if pos_tagged_ld else "UNK"

            clean_tokens_ld = model.to_tokens(prompt_ld)[:, -ctx:].to(device)
            with torch.no_grad():
                _, clean_cache_ld = model.run_with_cache(clean_tokens_ld)

            hm_ld_cpu = hm_ld.detach().cpu()
            pos_hm_matrices[pos_label_ld].append(hm_ld_cpu)

            nh = model.cfg.n_heads
            prev_ld = best_head_by_pos_normal_ld.get(pos_label_ld, [])
            best_head_by_pos_normal_ld[pos_label_ld] = _merge_top_heads_for_pos_margin_ratio(
                prev_ld, hm_ld, clean_cache_ld, nh, margin_ratio_heads_per_pos
            )

            flat_hm = hm_ld.flatten()
            top_k = min(margin_ratio_heads_per_pos, flat_hm.numel())
            topk_vals, topk_indices = torch.topk(flat_hm, top_k)

            step_top_heads = []
            for val, idx in zip(topk_vals, topk_indices):
                l, h = divmod(idx.item(), nh)
                step_top_heads.append({"layer": l, "head": h, "score": float(val.item())})

            token_level_results.append({
                "step": step,
                "token_str": last_token_str_ld,
                "token_id": last_token_id,
                "pos_tag": pos_label_ld,
                "top_heads": step_top_heads,
                "hm_matrix": hm_ld_cpu.tolist()
            })

            _, prompt_ld = generate_till_answer(model, prompt_ld, max_new_tokens=1)
            step += 1

        merged_ld_norm = {}
        for label, entries in best_head_by_pos_normal_ld.items():
            for info in entries:
                key = (info["layer"], info["head"])
                if (key not in merged_ld_norm) or (info["score"] > merged_ld_norm[key]["score"]):
                    merged_ld_norm[key] = {**info, "label": label}

        category_level_results = {}
        for pos_label, entries in best_head_by_pos_normal_ld.items():
            stacked_hms = torch.stack(pos_hm_matrices[pos_label])
            aggregated_hm_matrix = torch.mean(stacked_hms, dim=0)
            category_level_results[pos_label] = {
                "top_heads": [{"layer": info["layer"], "head": info["head"], "score": float(info["score"])} for info in entries],
                "aggregated_hm_matrix": aggregated_hm_matrix.tolist()
            }

        layer_to_specs = defaultdict(list)
        for (layer, head), info in merged_ld_norm.items():
            layer_to_specs[layer].append({"head": head, "vec": info["vec"]})

        def make_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = []
        for layer, specs in layer_to_specs.items():
            hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(specs)))

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits = patched_logits_full[0, -1, :]

        patched_t_true_logit = patched_logits[t_true].item()
        logit_increase = patched_t_true_logit - no_cot_t_true_logit

        patched_probs = F.softmax(patched_logits, dim=-1)
        patched_t_true_prob = patched_probs[t_true].item()
        prob_increase = patched_t_true_prob - no_cot_t_true_prob

        recovery_score = margin_recovery_ratio(clean_reference_logits, no_cot_logits, patched_logits).item()

        selected_heads_info = [
            {"layer": layer, "head": head, "pos_label": info["label"], "patch_score": float(info["score"])}
            for (layer, head), info in merged_ld_norm.items()
        ]

        nested_json_record = {
            "example_id": example_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_t_true_logit,
                "logit_increase": logit_increase,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": patched_t_true_prob,
                "prob_increase": prob_increase,
                "recovery_score": recovery_score
            },
            "patching_results": {
                "token_level": token_level_results,
                "category_level": category_level_results,
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> Target ID: {example_id} | Logit Inc: {logit_increase:.4f} | Prob Inc: {prob_increase:.4f} | Rec: {recovery_score:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


def multi_head_patching_with_jsd_metric(
    df,
    model,
    id_column="ID",
    ctx=1024,
    jsd_heads_per_pos=3,
    max_generation_steps=500,
    output_json_path="multi_head_patching_with_jsd_results.json"
):
    """
    Performs the multi-head patching experiment specifically for the JSD metric.
    Since JSD measures divergence, LOWER scores (closer to 0) are better.
    """
    metric_factory = make_jsd_metric 
    device = next(model.parameters()).device
    json_export_data = []

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Examples (JSD)"):
        example_id = row[id_column]
        cot_prompt_for_reference = row["PromptWithCot"]
        cot_prompt_for_patching = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]
        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(
            model, cot_prompt_for_reference
        )
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_probs = F.softmax(no_cot_logits, dim=-1)
        no_cot_t_true_prob = no_cot_probs[t_true].item()

        active_metric_factory = metric_factory(no_cot_logits)

        prompt_ld = cot_prompt_for_patching
        best_head_by_pos_normal_ld = {}
        pos_hm_matrices = defaultdict(list)
        step = 0
        token_level_results = []

        while not prompt_ld.endswith("The answer is ") and step < max_generation_steps:
            hm_ld = patching_pipeline(
                model,
                prompt_ld,
                no_cot_prompt_last_token,
                metric=active_metric_factory,
                patching_function=patch_attn_head_out_last_pos,
                clean_reference_logits=clean_reference_logits,
            )

            prompt_tokens_ld = model.to_tokens(prompt_ld)
            last_token_id = prompt_tokens_ld[0, -1].item()
            last_token_str_ld = model.to_string(prompt_tokens_ld[:, -1:])[0].strip()

            pos_tagged_ld = nltk.pos_tag([last_token_str_ld] if last_token_str_ld else ["."])
            pos_label_ld = pos_tagged_ld[0][1] if pos_tagged_ld else "UNK"

            clean_tokens_ld = model.to_tokens(prompt_ld)[:, -ctx:].to(device)
            with torch.no_grad():
                _, clean_cache_ld = model.run_with_cache(clean_tokens_ld)

            hm_ld_cpu = hm_ld.detach().cpu()
            pos_hm_matrices[pos_label_ld].append(hm_ld_cpu)

            nh = model.cfg.n_heads
            prev_ld = best_head_by_pos_normal_ld.get(pos_label_ld, [])

            # Select minimum values suitable for JSD
            best_head_by_pos_normal_ld[pos_label_ld] = _merge_top_heads_for_pos_jsd(
                prev_ld, hm_ld, clean_cache_ld, nh, jsd_heads_per_pos
            )

            flat_hm = hm_ld.flatten()
            top_k = min(jsd_heads_per_pos, flat_hm.numel())

            # Retrieve lowest values for JSD (largest=False)
            topk_vals, topk_indices = torch.topk(flat_hm, top_k, largest=False)

            step_top_heads = []
            for val, idx in zip(topk_vals, topk_indices):
                l, h = divmod(idx.item(), nh)
                step_top_heads.append({"layer": l, "head": h, "score": float(val.item())})

            token_level_results.append({
                "step": step,
                "token_str": last_token_str_ld,
                "token_id": last_token_id,
                "pos_tag": pos_label_ld,
                "top_heads": step_top_heads,
                "hm_matrix": hm_ld_cpu.tolist()
            })

            _, prompt_ld = generate_till_answer(model, prompt_ld, max_new_tokens=1)
            step += 1

        merged_ld_norm = {}
        for label, entries in best_head_by_pos_normal_ld.items():
            for info in entries:
                key = (info["layer"], info["head"])
                # Save the head if we found a lower (<) JSD score
                if (key not in merged_ld_norm) or (info["score"] < merged_ld_norm[key]["score"]):
                    merged_ld_norm[key] = {**info, "label": label}

        category_level_results = {}
        for pos_label, entries in best_head_by_pos_normal_ld.items():
            stacked_hms = torch.stack(pos_hm_matrices[pos_label])
            aggregated_hm_matrix = torch.mean(stacked_hms, dim=0)
            category_level_results[pos_label] = {
                "top_heads": [{"layer": info["layer"], "head": info["head"], "score": float(info["score"])} for info in entries],
                "aggregated_hm_matrix": aggregated_hm_matrix.tolist()
            }

        layer_to_specs = defaultdict(list)
        for (layer, head), info in merged_ld_norm.items():
            layer_to_specs[layer].append({"head": head, "vec": info["vec"]})

        def make_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = []
        for layer, specs in layer_to_specs.items():
            hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(specs)))

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits = patched_logits_full[0, -1, :]

        patched_t_true_logit = patched_logits[t_true].item()
        logit_increase = patched_t_true_logit - no_cot_t_true_logit

        patched_probs = F.softmax(patched_logits, dim=-1)
        patched_t_true_prob = patched_probs[t_true].item()
        prob_increase = patched_t_true_prob - no_cot_t_true_prob

        # Calculate Final JSD score instead of recovery score
        final_jsd_score = active_metric_factory(clean_reference_logits)(patched_logits).item()

        selected_heads_info = [
            {"layer": layer, "head": head, "pos_label": info["label"], "patch_score": float(info["score"])}
            for (layer, head), info in merged_ld_norm.items()
        ]

        nested_json_record = {
            "example_id": example_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_t_true_logit,
                "logit_increase": logit_increase,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": patched_t_true_prob,
                "prob_increase": prob_increase,
                "final_jsd_score": final_jsd_score
            },
            "patching_results": {
                "token_level": token_level_results,
                "category_level": category_level_results,
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> Target ID: {example_id} | Prob Inc: {prob_increase:.4f} | Final JSD: {final_jsd_score:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


# =============================================================================
# 2. Cross Patching Functions
# =============================================================================

def multi_head_cross_patching_with_margin_metric(
    df,
    model,
    id_column="ID", 
    ctx=1024,
    margin_ratio_heads_per_pos=3,
    max_generation_steps=500,
    output_json_path="multi_head_cross_patching_with_margin_results.json"
):
    """
    Performs the multi-head cross-patching experiment.
    - Reference CoT from current row.
    - Random CoT from a DIFFERENT row (based on ID) patched into the No-CoT prompt.
    """
    device = next(model.parameters()).device
    json_export_data = []
    metric_factory = make_margin_recovery_ratio_metric
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cross Patching (Margin)"):
        example_id = row[id_column]
        cot_prompt_for_reference = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]

        available_rows = df[df[id_column] != example_id]
        if available_rows.empty:
            raise ValueError("DataFrame needs at least 2 different IDs to perform cross-patching.")

        random_row = available_rows.sample(n=1).iloc[0]
        random_cot_prompt_for_patching = random_row["PromptWithCot"]
        random_source_id = random_row[id_column]

        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(
            model, cot_prompt_for_reference
        )
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_probs = F.softmax(no_cot_logits, dim=-1)
        no_cot_t_true_prob = no_cot_probs[t_true].item()

        active_metric_factory = metric_factory(no_cot_logits)

        # PATCHING LOOP
        prompt_ld = random_cot_prompt_for_patching
        best_head_by_pos_normal_ld = {}
        pos_hm_matrices = defaultdict(list)
        step = 0
        token_level_results = []

        while not prompt_ld.endswith("The answer is ") and step < max_generation_steps:
            hm_ld = patching_pipeline(
                model,
                prompt_ld,
                no_cot_prompt_last_token,
                metric=active_metric_factory,
                patching_function=patch_attn_head_out_last_pos,
                clean_reference_logits=clean_reference_logits,
            )

            prompt_tokens_ld = model.to_tokens(prompt_ld)
            last_token_id = prompt_tokens_ld[0, -1].item()
            last_token_str_ld = model.to_string(prompt_tokens_ld[:, -1:])[0].strip()

            pos_tagged_ld = nltk.pos_tag([last_token_str_ld] if last_token_str_ld else ["."])
            pos_label_ld = pos_tagged_ld[0][1] if pos_tagged_ld else "UNK"

            clean_tokens_ld = model.to_tokens(prompt_ld)[:, -ctx:].to(device)
            with torch.no_grad():
                _, clean_cache_ld = model.run_with_cache(clean_tokens_ld)

            hm_ld_cpu = hm_ld.detach().cpu()
            pos_hm_matrices[pos_label_ld].append(hm_ld_cpu)

            nh = model.cfg.n_heads
            prev_ld = best_head_by_pos_normal_ld.get(pos_label_ld, [])
            best_head_by_pos_normal_ld[pos_label_ld] = _merge_top_heads_for_pos_margin_ratio(
                prev_ld, hm_ld, clean_cache_ld, nh, margin_ratio_heads_per_pos
            )

            flat_hm = hm_ld.flatten()
            top_k = min(margin_ratio_heads_per_pos, flat_hm.numel())
            topk_vals, topk_indices = torch.topk(flat_hm, top_k)

            step_top_heads = []
            for val, idx in zip(topk_vals, topk_indices):
                l, h = divmod(idx.item(), nh)
                step_top_heads.append({"layer": l, "head": h, "score": float(val.item())})

            token_level_results.append({
                "step": step,
                "token_str": last_token_str_ld,
                "token_id": last_token_id,
                "pos_tag": pos_label_ld,
                "top_heads": step_top_heads,
                "hm_matrix": hm_ld_cpu.tolist()
            })

            _, prompt_ld = generate_till_answer(model, prompt_ld, max_new_tokens=1)
            step += 1

        merged_ld_norm = {}
        for label, entries in best_head_by_pos_normal_ld.items():
            for info in entries:
                key = (info["layer"], info["head"])
                if (key not in merged_ld_norm) or (info["score"] > merged_ld_norm[key]["score"]):
                    merged_ld_norm[key] = {**info, "label": label}

        category_level_results = {}
        for pos_label, entries in best_head_by_pos_normal_ld.items():
            stacked_hms = torch.stack(pos_hm_matrices[pos_label])
            aggregated_hm_matrix = torch.mean(stacked_hms, dim=0)
            category_level_results[pos_label] = {
                "top_heads": [{"layer": info["layer"], "head": info["head"], "score": float(info["score"])} for info in entries],
                "aggregated_hm_matrix": aggregated_hm_matrix.tolist()
            }

        # APPLY MULTI-HEAD PATCHING
        layer_to_specs = defaultdict(list)
        for (layer, head), info in merged_ld_norm.items():
            layer_to_specs[layer].append({"head": head, "vec": info["vec"]})

        def make_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = []
        for layer, specs in layer_to_specs.items():
            hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(specs)))

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits = patched_logits_full[0, -1, :]

        # METRICS
        patched_t_true_logit = patched_logits[t_true].item()
        logit_increase = patched_t_true_logit - no_cot_t_true_logit
        patched_probs = F.softmax(patched_logits, dim=-1)
        patched_t_true_prob = patched_probs[t_true].item()
        prob_increase = patched_t_true_prob - no_cot_t_true_prob
        recovery_score = margin_recovery_ratio(clean_reference_logits, no_cot_logits, patched_logits).item()

        selected_heads_info = [
            {"layer": layer, "head": head, "pos_label": info["label"], "patch_score": float(info["score"])}
            for (layer, head), info in merged_ld_norm.items()
        ]

        nested_json_record = {
            "example_id": example_id,
            "random_source_id": random_source_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_t_true_logit,
                "logit_increase": logit_increase,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": patched_t_true_prob,
                "prob_increase": prob_increase,
                "recovery_score": recovery_score
            },
            "patching_results": {
                "token_level": token_level_results,
                "category_level": category_level_results,
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> Target ID: {example_id} (Source: {random_source_id}) | Logit Inc: {logit_increase:.4f} | Rec: {recovery_score:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


def multi_head_cross_patching_with_jsd_metric(
    df,
    model,
    id_column="ID",
    ctx=1024,
    jsd_heads_per_pos=3,
    max_generation_steps=500,
    output_json_path="multi_head_cross_patching_with_jsd_results.json"
):
    """
    Performs the multi-head cross-patching experiment specifically for the JSD metric.
    LOWER scores (closer to 0) are better.
    """
    device = next(model.parameters()).device
    json_export_data = []
    metric_factory = make_jsd_metric

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cross Patching (JSD)"):
        example_id = row[id_column]
        cot_prompt_for_reference = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]

        available_rows = df[df[id_column] != example_id]
        if available_rows.empty:
            raise ValueError("DataFrame needs at least 2 different IDs to perform cross-patching.")

        random_row = available_rows.sample(n=1).iloc[0]
        random_cot_prompt_for_patching = random_row["PromptWithCot"]
        random_source_id = random_row[id_column]

        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(
            model, cot_prompt_for_reference
        )
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_probs = F.softmax(no_cot_logits, dim=-1)
        no_cot_t_true_prob = no_cot_probs[t_true].item()

        active_metric_factory = metric_factory(no_cot_logits)

        prompt_ld = random_cot_prompt_for_patching
        best_head_by_pos_normal_ld = {}
        pos_hm_matrices = defaultdict(list)
        step = 0
        token_level_results = []

        while not prompt_ld.endswith("The answer is ") and step < max_generation_steps:
            hm_ld = patching_pipeline(
                model,
                prompt_ld,
                no_cot_prompt_last_token,
                metric=active_metric_factory,
                patching_function=patch_attn_head_out_last_pos,
                clean_reference_logits=clean_reference_logits,
            )

            prompt_tokens_ld = model.to_tokens(prompt_ld)
            last_token_id = prompt_tokens_ld[0, -1].item()
            last_token_str_ld = model.to_string(prompt_tokens_ld[:, -1:])[0].strip()

            pos_tagged_ld = nltk.pos_tag([last_token_str_ld] if last_token_str_ld else ["."])
            pos_label_ld = pos_tagged_ld[0][1] if pos_tagged_ld else "UNK"

            clean_tokens_ld = model.to_tokens(prompt_ld)[:, -ctx:].to(device)
            with torch.no_grad():
                _, clean_cache_ld = model.run_with_cache(clean_tokens_ld)

            hm_ld_cpu = hm_ld.detach().cpu()
            pos_hm_matrices[pos_label_ld].append(hm_ld_cpu)

            nh = model.cfg.n_heads
            prev_ld = best_head_by_pos_normal_ld.get(pos_label_ld, [])

            best_head_by_pos_normal_ld[pos_label_ld] = _merge_top_heads_for_pos_jsd(
                prev_ld, hm_ld, clean_cache_ld, nh, jsd_heads_per_pos
            )

            flat_hm = hm_ld.flatten()
            top_k = min(jsd_heads_per_pos, flat_hm.numel())
            topk_vals, topk_indices = torch.topk(flat_hm, top_k, largest=False)

            step_top_heads = []
            for val, idx in zip(topk_vals, topk_indices):
                l, h = divmod(idx.item(), nh)
                step_top_heads.append({"layer": l, "head": h, "score": float(val.item())})

            token_level_results.append({
                "step": step,
                "token_str": last_token_str_ld,
                "token_id": last_token_id,
                "pos_tag": pos_label_ld,
                "top_heads": step_top_heads,
                "hm_matrix": hm_ld_cpu.tolist()
            })

            _, prompt_ld = generate_till_answer(model, prompt_ld, max_new_tokens=1)
            step += 1

        merged_ld_norm = {}
        for label, entries in best_head_by_pos_normal_ld.items():
            for info in entries:
                key = (info["layer"], info["head"])
                if (key not in merged_ld_norm) or (info["score"] < merged_ld_norm[key]["score"]):
                    merged_ld_norm[key] = {**info, "label": label}

        category_level_results = {}
        for pos_label, entries in best_head_by_pos_normal_ld.items():
            stacked_hms = torch.stack(pos_hm_matrices[pos_label])
            aggregated_hm_matrix = torch.mean(stacked_hms, dim=0)
            category_level_results[pos_label] = {
                "top_heads": [{"layer": info["layer"], "head": info["head"], "score": float(info["score"])} for info in entries],
                "aggregated_hm_matrix": aggregated_hm_matrix.tolist()
            }

        layer_to_specs = defaultdict(list)
        for (layer, head), info in merged_ld_norm.items():
            layer_to_specs[layer].append({"head": head, "vec": info["vec"]})

        def make_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = []
        for layer, specs in layer_to_specs.items():
            hooks.append((f"blocks.{layer}.attn.hook_z", make_hook(specs)))

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits = patched_logits_full[0, -1, :]

        patched_t_true_logit = patched_logits[t_true].item()
        logit_increase = patched_t_true_logit - no_cot_t_true_logit
        patched_probs = F.softmax(patched_logits, dim=-1)
        patched_t_true_prob = patched_probs[t_true].item()
        prob_increase = patched_t_true_prob - no_cot_t_true_prob

        final_jsd_score = active_metric_factory(clean_reference_logits)(patched_logits).item()

        selected_heads_info = [
            {"layer": layer, "head": head, "pos_label": info["label"], "patch_score": float(info["score"])}
            for (layer, head), info in merged_ld_norm.items()
        ]

        nested_json_record = {
            "example_id": example_id,
            "random_source_id": random_source_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_t_true_logit,
                "logit_increase": logit_increase,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": patched_t_true_prob,
                "prob_increase": prob_increase,
                "final_jsd_score": final_jsd_score
            },
            "patching_results": {
                "token_level": token_level_results,
                "category_level": category_level_results,
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> Target ID: {example_id} | Prob Inc: {prob_increase:.4f} | Final JSD: {final_jsd_score:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


# =============================================================================
# 3. Random Patching Control Functions
# =============================================================================

def sequential_random_patching_margin(
    df,
    model,
    id_column="ID",
    ctx=1024,
    margin_ratio_heads_per_pos=3, 
    reference_json_path="multi_head_patching_with_margin_results.json",
    output_json_path="random_patching_margin_results.json"
):
    """
    Algorithm 4: Random Activation Patching (Margin Recovery Metric)
    Serves as a control experiment by measuring the effect of injecting random Gaussian noise.
    Dynamically matches the number of patched heads per example from the reference experiment.
    """
    device = next(model.parameters()).device
    json_export_data = []

    with open(reference_json_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    reference_heads_map = {
        item["example_id"]: item["patching_results"]["final_multi_head"]["num_heads_patched"]
        for item in ref_data if "example_id" in item
    }

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_model // n_heads

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Random Control (Margin)"):
        example_id = row[id_column]
        cot_prompt = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]
        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(model, cot_prompt)
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_t_true_prob = F.softmax(no_cot_logits, dim=-1)[t_true].item()

        # Sequential Scan
        r_rand = []
        z_table = {}

        for l in range(n_layers):
            for h in range(n_heads):
                z_rand = torch.randn(d_head).to(device)
                z_table[(l, h)] = z_rand

                def rand_hook(value, hook, layer=l, head=h, vec=z_rand):
                    v = value.clone()
                    v[:, v.shape[1]-1, head, :] = vec
                    return v

                with torch.no_grad():
                    patched_logits_single = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(f"blocks.{l}.attn.hook_z", rand_hook)],
                        return_type="logits"
                    )[0, -1, :]

                rec_single = margin_recovery_ratio(clean_reference_logits, no_cot_logits, patched_logits_single).item()
                r_rand.append({"layer": l, "head": h, "score": rec_single, "mrr": 0.0})

        r_rand.sort(key=lambda x: x["score"], reverse=True)
        n_heads_to_patch = reference_heads_map.get(example_id, margin_ratio_heads_per_pos)
        h_topn = r_rand[:n_heads_to_patch]

        # Joint Patching
        layer_to_specs = defaultdict(list)
        for info in h_topn:
            l, h = info["layer"], info["head"]
            layer_to_specs[l].append({"head": h, "vec": z_table[(l, h)]})

        def make_multi_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = [(f"blocks.{layer}.attn.hook_z", make_multi_hook(specs))
                 for layer, specs in layer_to_specs.items()]

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits_topn = patched_logits_full[0, -1, :]

        final_recovery_score = margin_recovery_ratio(clean_reference_logits, no_cot_logits, patched_logits_topn).item()
        patched_t_true_logit = patched_logits_topn[t_true].item()
        patched_t_true_prob = F.softmax(patched_logits_topn, dim=-1)[t_true].item()

        selected_heads_info = [
            {"layer": x["layer"], "head": x["head"], "pos_label": "RANDOM", "patch_score": float(x["score"])}
            for x in h_topn
        ]

        nested_json_record = {
            "example_id": example_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_t_true_logit,
                "logit_increase": patched_t_true_logit - no_cot_t_true_logit,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": patched_t_true_prob,
                "prob_increase": patched_t_true_prob - no_cot_t_true_prob,
                "recovery_score": final_recovery_score
            },
            "patching_results": {
                "token_level": [],
                "category_level": {},
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> RANDOM (Margin) ID {example_id} | Final Rec Score: {final_recovery_score:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


def sequential_random_patching_jsd(
    df,
    model,
    id_column="ID",
    ctx=1024,
    jsd_heads_per_pos=3, 
    reference_json_path="multi_head_patching_with_jsd_results.json",
    output_json_path="random_patching_jsd_results.json"
):
    """
    Algorithm 4: Random Activation Patching (JSD Metric)
    Serves as a control experiment by measuring the effect of injecting random Gaussian noise.
    Dynamically matches the number of patched heads per example from the reference experiment.
    """
    metric_factory = make_jsd_metric
    device = next(model.parameters()).device
    json_export_data = []

    with open(reference_json_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    reference_heads_map = {
        item["example_id"]: item["patching_results"]["final_multi_head"]["num_heads_patched"]
        for item in ref_data if "example_id" in item
    }

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_model // n_heads

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Random Control (JSD)"):
        example_id = row[id_column]
        cot_prompt = row["PromptWithCot"]
        no_cot_prompt = row["PromptWithoutCot"]
        no_cot_prompt_last_token = no_cot_prompt + " The answer is "

        full_answer_text, clean_reference_logits = generate_full_answer_and_get_logits(model, cot_prompt)
        t_true = int(clean_reference_logits.argmax(dim=-1).item())

        corrupted_tokens = model.to_tokens(no_cot_prompt_last_token)[:, -ctx:].to(device)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)
            no_cot_logits = corrupted_logits[0, -1, :]

        no_cot_t_true_logit = no_cot_logits[t_true].item()
        no_cot_t_true_prob = F.softmax(no_cot_logits, dim=-1)[t_true].item()
        active_metric_factory = metric_factory(no_cot_logits)

        jsd_base = active_metric_factory(clean_reference_logits)(no_cot_logits).item()

        r_rand = []
        z_table = {}

        for l in range(n_layers):
            for h in range(n_heads):
                z_rand = torch.randn(d_head).to(device)
                z_table[(l, h)] = z_rand

                def rand_hook(value, hook, layer=l, head=h, vec=z_rand):
                    v = value.clone()
                    v[:, v.shape[1]-1, head, :] = vec
                    return v

                with torch.no_grad():
                    patched_logits_single = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(f"blocks.{l}.attn.hook_z", rand_hook)],
                        return_type="logits"
                    )[0, -1, :]

                jsd_single = active_metric_factory(clean_reference_logits)(patched_logits_single).item()
                r_rand.append({"layer": l, "head": h, "score": jsd_single, "mrr": 0.0})

        r_rand.sort(key=lambda x: x["score"])

        n_heads_to_patch = reference_heads_map.get(example_id, jsd_heads_per_pos)
        h_topn = r_rand[:n_heads_to_patch]

        layer_to_specs = defaultdict(list)
        for info in h_topn:
            l, h = info["layer"], info["head"]
            layer_to_specs[l].append({"head": h, "vec": z_table[(l, h)]})

        def make_multi_hook(specs):
            def hook_fn(value, hook):
                v = value.clone()
                last_pos = v.shape[1] - 1
                for spec in specs:
                    v[:, last_pos, spec["head"], :] = spec["vec"].to(v.device, v.dtype)
                return v
            return hook_fn

        hooks = [(f"blocks.{layer}.attn.hook_z", make_multi_hook(specs))
                 for layer, specs in layer_to_specs.items()]

        with torch.no_grad():
            patched_logits_full = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks, return_type="logits")
        patched_logits_topn = patched_logits_full[0, -1, :]

        final_jsd_topn = active_metric_factory(clean_reference_logits)(patched_logits_topn).item()
        prob_inc = F.softmax(patched_logits_topn, dim=-1)[t_true].item() - no_cot_t_true_prob

        selected_heads_info = [
            {"layer": x["layer"], "head": x["head"], "pos_label": "RANDOM", "patch_score": float(x["score"])}
            for x in h_topn
        ]

        nested_json_record = {
            "example_id": example_id,
            "t_true_token_id": t_true,
            "metrics": {
                "no_cot_logit": no_cot_t_true_logit,
                "patched_logit": patched_logits_topn[t_true].item(),
                "logit_increase": patched_logits_topn[t_true].item() - no_cot_t_true_logit,
                "no_cot_prob": no_cot_t_true_prob,
                "patched_prob": F.softmax(patched_logits_topn, dim=-1)[t_true].item(),
                "prob_increase": prob_inc,
                "final_jsd_score": final_jsd_topn,
                "baseline_jsd": jsd_base
            },
            "patching_results": {
                "token_level": [],
                "category_level": {},
                "final_multi_head": {
                    "num_heads_patched": len(selected_heads_info),
                    "selected_heads": selected_heads_info
                }
            }
        }
        json_export_data.append(nested_json_record)
        tqdm.write(f"-> RANDOM ID {example_id} | JSD Baseline: {jsd_base:.4f} | JSD Final: {final_jsd_topn:.4f}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_export_data, f, indent=4, ensure_ascii=False)

    return json_export_data


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    
    print("Loading Model...")
    model = load_model("qwen2.5-0.5b", device="cuda")
    
    # Load the curated SVAMP dataset from data/processed folder
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'svamp_curated_subset.json'))
    print(f"Loading Dataset from {data_path}...")
    
    try:
        sampled_df = pd.read_json(data_path)
        print(f"Successfully loaded {len(sampled_df)} examples.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find dataset at: {data_path}")
        print("Make sure the 'data/processed' folder exists in your project root and contains 'svamp_curated_subset.json'.")
        sys.exit(1)

    print("==================================================")
    print("1. EXPERIMENT: Normal Patching (Margin Difference)")
    print("==================================================")
    json_results_1 = multi_head_patching_with_margin_difference(
        df=sampled_df, 
        model=model, 
        id_column="ID", 
        output_json_path="multi_head_patching_with_margin_results.json"
    )

    print("\n==================================================")
    print("2. EXPERIMENT: Cross Patching (Margin Difference)")
    print("==================================================")
    if len(sampled_df) > 1:
        json_results_2 = multi_head_cross_patching_with_margin_metric(
            df=sampled_df, 
            model=model, 
            id_column="ID",
            output_json_path="multi_head_cross_patching_with_margin_results.json"
        )
    else:
        print("Skipping Cross Patching (requires at least 2 rows).")

    print("\n==================================================")
    print("3. EXPERIMENT: Normal Patching (JSD)")
    print("==================================================")
    json_results_3 = multi_head_patching_with_jsd_metric(
        df=sampled_df, 
        model=model, 
        id_column="ID",
        output_json_path="multi_head_patching_with_jsd_results.json"
    )

    print("\n==================================================")
    print("4. EXPERIMENT: Cross Patching (JSD)")
    print("==================================================")
    if len(sampled_df) > 1:
        json_results_4 = multi_head_cross_patching_with_jsd_metric(
            df=sampled_df, 
            model=model, 
            id_column="ID",
            output_json_path="multi_head_cross_patching_with_jsd_results.json"
        )
    else:
        print("Skipping Cross Patching JSD (requires at least 2 rows).")

    print("\n==================================================")
    print("5. CONTROL EXPERIMENT: Random Patching (Margin)")
    print("==================================================")
    margin_results = sequential_random_patching_margin(
        df=sampled_df,
        model=model,
        id_column="ID",                           
        ctx=2048,                                 
        margin_ratio_heads_per_pos=3,             
        reference_json_path="multi_head_patching_with_margin_results.json", 
        output_json_path="random_patching_margin_results.json"              
    )
    print("\n[SUCCESS] Margin control experiment completed and saved to JSON.\n")

    print("==================================================")
    print("6. CONTROL EXPERIMENT: Random Patching (JSD)")
    print("==================================================")
    jsd_results = sequential_random_patching_jsd(
        df=sampled_df,
        model=model,
        id_column="ID",
        ctx=2048,
        jsd_heads_per_pos=3,
        reference_json_path="multi_head_patching_with_jsd_results.json", 
        output_json_path="random_patching_jsd_results.json"              
    )
    print("\n[SUCCESS] JSD control experiment completed and saved to JSON.")
    print("==================================================")