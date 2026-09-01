"""
src/ablation.py

Zero-ablation verification experiments.

Two prompt conditions are ablated: No-CoT and CoT. Each is scored three ways --
unablated, with the selected heads zeroed, and with a random head set of the
same size zeroed (the control).

Generation runs with the ablation hooks attached for the whole trajectory rather
than re-applied per token, so the heads stay silent across the entire reasoning
chain instead of only at the answer position.
"""

import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from src.tasks import get_task
from src.templates import get_template
from src.utils import (
    _append_token,
    _decode,
    _decode_generated_only,
    _decode_single_token,
    build_heads_by_example_id_from_curated,
    get_example_id_from_row,
    get_type_from_row,
    make_cot_prompt_from_row,
    make_nocot_prompt_from_row,
    make_zero_ablation_hooks,
    sample_random_heads_same_count_for_example,
)


# ============================================================
# Checkpointing (per-example JSONL, mirrors run_patchings.py's)
# ============================================================

def _load_ablation_checkpoint(path):
    """Reads a JSONL checkpoint into {example_id: row_dict}."""
    done = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[str(rec["example_id"])] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _append_ablation_checkpoint(path, record):
    """
    Appends and flushes one row. default=str is defensive: a couple of the
    row's fields (the gold answer, in particular) can come straight out of a
    pandas cell as a numpy scalar, which json.dumps otherwise rejects.
    """
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# Generation under ablation
# ============================================================

def _last_position_logits(model, tokens):
    """Run the transformer but unembed only its final sequence position.

    ``HookedTransformer.forward`` normally unembeds every position into the
    full vocabulary.  Autoregressive generation only reads the final position,
    and Qwen's 152k-token vocabulary makes the unused logits increasingly large
    as the generated trace grows.  Stopping after the last transformer block,
    slicing the residual, then applying the same final norm and unembed is
    exactly equivalent because both operations are position-wise.
    """
    residual = model(
        tokens,
        return_type=None,
        stop_at_layer=model.cfg.n_layers,
    )
    final_residual = residual[:, -1:, :]
    if model.cfg.normalization_type is not None:
        final_residual = model.ln_final(final_residual)
    logits = model.unembed(final_residual)
    if model.cfg.output_logits_soft_cap > 0.0:
        logits = model.cfg.output_logits_soft_cap * torch.tanh(
            logits / model.cfg.output_logits_soft_cap
        )
    return logits[0, -1, :]


def _generate_with_ablation(
    model,
    prompt,
    ablated_heads=None,
    max_new_tokens=1024,
    context_window=2048,
    prepend_bos=True,
    print_tokens=False,
    task=None,
):
    """
    Greedy generation with the given heads zeroed for the whole trajectory.

    Runs in two phases:
      1. generate until the task's answer trigger appears
      2. keep generating while the tokens still belong to the answer

    Phase 2 is what makes multi-token answers work: 92 arrives as "9" + "2", and
    True can arrive as "Tr" + "ue". The task owns that decision, because digits
    and words need different gates.

    Returns the generated text only, never the prompt, so no BOS handling is
    needed downstream.
    """
    task = task or get_task()
    device = next(model.parameters()).device
    model.reset_hooks()

    input_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    output_tokens = input_tokens.clone()
    hooks = make_zero_ablation_hooks(ablated_heads) if ablated_heads else []

    def _loop():
        nonlocal output_tokens
        generated = 0

        # A previous trajectory may leave large, differently-sized cached CUDA
        # blocks behind.  Releasing only unused blocks here preserves tensors
        # and results while giving the next trajectory a compact starting point.
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        with torch.inference_mode():
            # phase 1 -- reason until the answer trigger
            current_text = _decode(model, output_tokens)
            while not task.ends_reasoning(current_text) and generated < max_new_tokens:
                logits = _last_position_logits(model, output_tokens[:, -context_window:])
                next_token = logits.argmax(dim=-1, keepdim=True)
                output_tokens = _append_token(output_tokens, next_token)
                del logits
                generated += 1
                if generated % 64 == 0 and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                if print_tokens:
                    print(generated, repr(_decode_single_token(model, next_token)))
                current_text = _decode(model, output_tokens)

            # the trigger never appeared: hand back whatever was produced
            if not task.ends_reasoning(current_text):
                return _decode_generated_only(model, input_tokens, output_tokens)

            # phase 2 -- collect the answer tokens
            while generated < max_new_tokens:
                logits = _last_position_logits(model, output_tokens[:, -context_window:])
                next_token = logits.argmax(dim=-1, keepdim=True)
                del logits
                token_str = _decode_single_token(model, next_token)
                if not task.is_answer_continuation(token_str):
                    break
                output_tokens = _append_token(output_tokens, next_token)
                generated += 1
                if generated % 64 == 0 and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                if print_tokens:
                    print(generated, repr(token_str))

        return _decode_generated_only(model, input_tokens, output_tokens)

    if hooks:
        # a single context for the whole trajectory keeps the heads ablated throughout
        with model.hooks(fwd_hooks=hooks):
            return _loop()
    return _loop()


def generate_with_optional_ablation(
    model, prompt, ablated_heads=None, max_new_tokens=1024, print_tokens=False, task=None
):
    """No-CoT condition: the prompt already ends at the answer trigger."""
    return _generate_with_ablation(
        model, prompt, ablated_heads=ablated_heads, max_new_tokens=max_new_tokens,
        print_tokens=print_tokens, task=task,
    )


def generate_cot_with_optional_ablation(
    model, prompt, ablated_heads=None, max_cot_reasoning_tokens=1024,
    max_answer_tokens=None, print_tokens=False, task=None,
):
    """
    CoT condition: the model reasons first, then answers. max_answer_tokens is
    accepted for call-site compatibility; the answer length is governed by the
    task's continuation gate rather than by a separate budget.
    """
    return _generate_with_ablation(
        model, prompt, ablated_heads=ablated_heads, max_new_tokens=max_cot_reasoning_tokens,
        print_tokens=print_tokens, task=task,
    )


# ============================================================
# 7. NO-COT ABLATION
# ============================================================

def run_nocot_ablation_using_curated_heads(
    sampled_df,
    model,
    curated_normal_results,
    selected_heads_key="selected_heads",
    max_examples=None,
    max_new_tokens=2048,
    random_seed=42,
    print_tokens=False,
    verbose=False,
    task=None,
    template=None,
    checkpoint_path=None,
):
    """
    Her sampled_df örneği için:

        1. No-CoT prompt oluşturulur.
           Örn:
               "... The answer is "

        2. Normal No-CoT üretim yapılır.

        3. Selected heads ablate edilerek No-CoT üretim yapılır.

        4. Aynı sayıda random heads ablate edilerek kontrol üretimi yapılır.
    """
    task = task or get_task()
    template = template or get_template()

    heads_by_example_id = build_heads_by_example_id_from_curated(
        curated_normal_results,
        selected_heads_key
    )

    results = []
    done = _load_ablation_checkpoint(checkpoint_path)

    total = len(sampled_df) if max_examples is None else min(len(sampled_df), max_examples)
    pbar = tqdm(total=total, desc="Running No-CoT Ablation")

    count = 0

    for idx, row in sampled_df.iterrows():
        if max_examples is not None and count >= max_examples:
            break

        example_id = get_example_id_from_row(row, idx)
        cached = done.get(str(example_id))
        if cached is not None:
            results.append(cached)
            count += 1
            pbar.update(1)
            continue

        true_ans = task.gold_from_row(row)
        q_type = get_type_from_row(row)

        nocot_prompt = make_nocot_prompt_from_row(row, template=template, task=task)

        selected_heads_list = heads_by_example_id.get(example_id, [])
        num_heads = len(selected_heads_list)

        # ------------------------------
        # Normal generation
        # ------------------------------
        normal_text = generate_with_optional_ablation(
            model,
            nocot_prompt,
            ablated_heads=None,
            max_new_tokens=max_new_tokens,
            print_tokens=print_tokens,
            task=task,
        )

        # If the unablated run never reached the trigger there is no baseline to
        # ablate away, so the row is marked and excluded from accuracy rather than
        # counted as a failure caused by the ablation.
        skipped = task.answer_trigger not in (normal_text or "")
        normal_extracted = task.extract(normal_text)
        normal_correct = task.answers_equal(normal_extracted, true_ans)

        # ------------------------------
        # Selected-head ablation
        # ------------------------------
        ablation_text = None
        ablation_extracted = None
        ablation_correct = None

        if num_heads > 0:
            ablation_text = generate_with_optional_ablation(
                model,
                nocot_prompt,
                ablated_heads=selected_heads_list,
                max_new_tokens=max_new_tokens,
                print_tokens=print_tokens,
                task=task,
            )

            ablation_extracted = task.extract(ablation_text)
            ablation_correct = task.answers_equal(ablation_extracted, true_ans)

        # ------------------------------
        # Random-head ablation
        # ------------------------------
        random_text = None
        random_extracted = None
        random_correct = None
        random_heads_list = None

        if num_heads > 0:
            random_heads_list = sample_random_heads_same_count_for_example(
                model,
                num_heads,
                seed=random_seed + count
            )

            random_text = generate_with_optional_ablation(
                model,
                nocot_prompt,
                ablated_heads=random_heads_list,
                max_new_tokens=max_new_tokens,
                print_tokens=print_tokens,
                task=task,
            )

            random_extracted = task.extract(random_text)
            random_correct = task.answers_equal(random_extracted, true_ans)

        record = {
            "example_id": example_id,
            "Type": q_type,
            "skipped": skipped,
            "NoCotPrompt": nocot_prompt,
            "true_answer": true_ans,
            "selected_heads": selected_heads_list,
            "random_heads": random_heads_list,
            "num_heads_ablated": num_heads,

            "normal_text": normal_text,
            "normal_extracted": normal_extracted,
            "normal_correct": normal_correct,

            "ablation_text": ablation_text,
            "ablation_extracted": ablation_extracted,
            "ablation_correct": ablation_correct,

            "random_text": random_text,
            "random_extracted": random_extracted,
            "random_correct": random_correct,

            "correct": ablation_correct if ablation_correct is not None else False
        }
        results.append(record)
        _append_ablation_checkpoint(checkpoint_path, record)

        count += 1
        pbar.update(1)

        if verbose:
            print("=" * 80)
            print("example_id:", example_id)
            print("prompt:", repr(nocot_prompt))
            print("true:", true_ans)
            print("normal:", repr(normal_text), normal_extracted, normal_correct)
            print("ablated:", repr(ablation_text), ablation_extracted, ablation_correct)
            print("random:", repr(random_text), random_extracted, random_correct)

    pbar.close()
    return pd.DataFrame(results)


# ============================================================
# 8. COT ABLATION
# ============================================================

def run_cot_ablation_using_curated_heads(
    sampled_df,
    model,
    curated_normal_results,
    selected_heads_key="selected_heads",
    max_examples=None,
    max_cot_reasoning_tokens=2048,
    max_answer_tokens=None,
    random_seed=42,
    print_tokens=False,
    verbose=False,
    task=None,
    template=None,
    checkpoint_path=None,
):
    """
    Her sampled_df örneği için:

        1. CoT prompt oluşturulur.
           Örn:
               "... Let's think step by step."

        2. Model token token reasoning üretir.

        3. "The answer is " görünce numeric answer tamamlanır.

        4. Selected heads ablation ve random heads control aynı generation loop içinde yapılır.
    """
    task = task or get_task()
    template = template or get_template()

    heads_by_example_id = build_heads_by_example_id_from_curated(
        curated_normal_results,
        selected_heads_key
    )

    results = []
    done = _load_ablation_checkpoint(checkpoint_path)

    total = len(sampled_df) if max_examples is None else min(len(sampled_df), max_examples)
    pbar = tqdm(total=total, desc="Running CoT Ablation")

    count = 0

    for idx, row in sampled_df.iterrows():
        if max_examples is not None and count >= max_examples:
            break

        example_id = get_example_id_from_row(row, idx)
        cached = done.get(str(example_id))
        if cached is not None:
            results.append(cached)
            count += 1
            pbar.update(1)
            continue

        true_ans = task.gold_from_row(row)
        q_type = get_type_from_row(row)

        cot_prompt = make_cot_prompt_from_row(row, template=template)

        selected_heads_list = heads_by_example_id.get(example_id, [])
        num_heads = len(selected_heads_list)

        # ------------------------------
        # Normal generation
        # ------------------------------
        normal_text = generate_cot_with_optional_ablation(
            model,
            cot_prompt,
            ablated_heads=None,
            max_cot_reasoning_tokens=max_cot_reasoning_tokens,
            max_answer_tokens=max_answer_tokens,
            print_tokens=print_tokens,
            task=task,
        )

        # If the unablated run never reached the trigger there is no baseline to
        # ablate away, so the row is marked and excluded from accuracy rather than
        # counted as a failure caused by the ablation.
        skipped = task.answer_trigger not in (normal_text or "")
        normal_extracted = task.extract(normal_text)
        normal_correct = task.answers_equal(normal_extracted, true_ans)

        # ------------------------------
        # Selected-head ablation
        # ------------------------------
        ablation_text = None
        ablation_extracted = None
        ablation_correct = None

        if num_heads > 0:
            ablation_text = generate_cot_with_optional_ablation(
                model,
                cot_prompt,
                ablated_heads=selected_heads_list,
                max_cot_reasoning_tokens=max_cot_reasoning_tokens,
                max_answer_tokens=max_answer_tokens,
                print_tokens=print_tokens,
                task=task,
            )

            ablation_extracted = task.extract(ablation_text)
            ablation_correct = task.answers_equal(ablation_extracted, true_ans)

        # ------------------------------
        # Random-head ablation
        # ------------------------------
        random_text = None
        random_extracted = None
        random_correct = None
        random_heads_list = None

        if num_heads > 0:
            random_heads_list = sample_random_heads_same_count_for_example(
                model,
                num_heads,
                seed=random_seed + count
            )

            random_text = generate_cot_with_optional_ablation(
                model,
                cot_prompt,
                ablated_heads=random_heads_list,
                max_cot_reasoning_tokens=max_cot_reasoning_tokens,
                max_answer_tokens=max_answer_tokens,
                print_tokens=print_tokens,
                task=task,
            )

            random_extracted = task.extract(random_text)
            random_correct = task.answers_equal(random_extracted, true_ans)

        record = {
            "example_id": example_id,
            "Type": q_type,
            "skipped": skipped,
            "CotPrompt": cot_prompt,
            "true_answer": true_ans,
            "selected_heads": selected_heads_list,
            "random_heads": random_heads_list,
            "num_heads_ablated": num_heads,

            "normal_text": normal_text,
            "normal_extracted": normal_extracted,
            "normal_correct": normal_correct,

            "ablation_text": ablation_text,
            "ablation_extracted": ablation_extracted,
            "ablation_correct": ablation_correct,

            "random_text": random_text,
            "random_extracted": random_extracted,
            "random_correct": random_correct,

            "correct": ablation_correct if ablation_correct is not None else False
        }
        results.append(record)
        _append_ablation_checkpoint(checkpoint_path, record)

        count += 1
        pbar.update(1)

        if verbose:
            print("=" * 80)
            print("example_id:", example_id)
            print("prompt:", repr(cot_prompt))
            print("true:", true_ans)
            print("normal:", repr(normal_text), normal_extracted, normal_correct)
            print("ablated:", repr(ablation_text), ablation_extracted, ablation_correct)
            print("random:", repr(random_text), random_extracted, random_correct)

    pbar.close()
    return pd.DataFrame(results)

def load_heads_from_experiment(file_path):
    """
    JSON dosyasındaki selected_heads listesini şu formata dönüştürür:

    [
        {
            "example_id": "...",
            "selected_heads": [
                {"layer": 3, "head": 5},
                {"layer": 7, "head": 2},
                ...
            ]
        },
        ...
    ]

    Beklenen JSON yapısı:
        item["patching_results"]["final_multi_head"]["selected_heads"]

    Eğer JSON dosyası bozuk/kesikse regex fallback ile kurtarmaya çalışır.
    """

    heads_by_example_id = {}

    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return []

    # ------------------------------------------------------------
    # Normal JSON okuma
    # ------------------------------------------------------------
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            eid = item.get("example_id")

            if eid is None:
                continue

            selected = (
                item.get("patching_results", {})
                    .get("final_multi_head", {})
                    .get("selected_heads", [])
            )

            heads = []

            for h in selected:
                if h is None:
                    continue

                if "layer" in h and "head" in h:
                    heads.append({
                        "layer": int(h["layer"]),
                        "head": int(h["head"])
                    })

            heads_by_example_id[str(eid)] = heads

    # ------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------
    except Exception as e:
        print(f"⚠️ JSON normal okunamadı, regex fallback deneniyor: {file_path}")
        print(f"Hata: {e}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split('"example_id":')[1:]

        for block in blocks:
            eid_match = re.search(r'\s*"([^"]+)"', block)

            if not eid_match:
                continue

            eid = eid_match.group(1)

            heads = []

            selected_block = block.split('"selected_heads":')

            if len(selected_block) > 1:
                heads_part = selected_block[1].split("]")[0]

                layer_matches = re.findall(r'"layer":\s*(\d+)', heads_part)
                head_matches = re.findall(r'"head":\s*(\d+)', heads_part)

                for layer, head in zip(layer_matches, head_matches):
                    heads.append({
                        "layer": int(layer),
                        "head": int(head)
                    })

            heads_by_example_id[str(eid)] = heads

    return [
        {
            "example_id": eid,
            "selected_heads": heads
        }
        for eid, heads in heads_by_example_id.items()
    ]
