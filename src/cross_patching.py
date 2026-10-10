"""Cross-example sequential activation patching.

The donor example A supplies the CoT trajectory and cached head activations.
Those activations are patched into the final position of target example B's
No-CoT prompt.  Scores always use B's clean CoT logits as the target and B's
No-CoT logits as the baseline.
"""

import itertools
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _stable_seed(seed, *parts):
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return (int(seed) + zlib.crc32(payload)) % (2**31 - 1)


def build_balanced_cross_pairs(
    df,
    target_n=32,
    seed=42,
    id_column="ID",
    answer_column="Answer",
    strata=("OperationCount", "Type"),
):
    """Select a balanced target set and create a one-to-one donor derangement.

    Donors come from the same 32-question set, match the target's SVAMP
    operation stratum, are never the target itself, and have a different gold
    answer.  Within each stratum every selected question is used exactly once
    as a target and once as a donor.
    """
    required = {id_column, answer_column, *strata}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing columns for cross pairing: {missing}")
    if df[id_column].astype(str).duplicated().any():
        raise ValueError(f"{id_column} values must be unique")

    groups = list(df.groupby(list(strata), sort=True, dropna=False))
    if not groups or target_n % len(groups):
        raise ValueError(
            f"target_n={target_n} must be divisible by the {len(groups)} strata"
        )
    per_group = target_n // len(groups)
    selected_groups = []
    pairs = []

    for key, group in groups:
        if len(group) < per_group:
            raise ValueError(f"stratum {key!r} has {len(group)} rows; need {per_group}")
        picked = group.sample(
            n=per_group,
            random_state=_stable_seed(seed, "sample", *key),
        ).copy()
        picked[id_column] = picked[id_column].astype(str)
        selected_groups.append(picked)

        records = picked.to_dict("records")
        assignments = list(itertools.permutations(range(per_group)))
        random.Random(_stable_seed(seed, "donors", *key)).shuffle(assignments)
        assignment = next(
            (
                perm
                for perm in assignments
                if all(
                    i != j
                    and str(records[i][answer_column]) != str(records[j][answer_column])
                    for i, j in enumerate(perm)
                )
            ),
            None,
        )
        if assignment is None:
            raise ValueError(
                f"stratum {key!r} has no one-to-one donor assignment with different answers"
            )

        for target_idx, source_idx in enumerate(assignment):
            target = records[target_idx]
            source = records[source_idx]
            pair = {
                "target_id": str(target[id_column]),
                "source_id": str(source[id_column]),
                "target_answer": str(target[answer_column]),
                "source_answer": str(source[answer_column]),
            }
            pair.update({name: target[name] for name in strata})
            pairs.append(pair)

    selected = pd.concat(selected_groups, ignore_index=True)
    selected[id_column] = selected[id_column].astype(str)
    validate_cross_pairs(pairs, target_n=target_n, strata=strata)
    return selected, pairs


def validate_cross_pairs(pairs, target_n=32, strata=("OperationCount", "Type")):
    if len(pairs) != target_n:
        raise ValueError(f"expected {target_n} pairs, found {len(pairs)}")
    targets = [str(p["target_id"]) for p in pairs]
    sources = [str(p["source_id"]) for p in pairs]
    if len(set(targets)) != target_n or len(set(sources)) != target_n:
        raise ValueError("targets and donors must each be unique")
    if set(targets) != set(sources):
        raise ValueError("donors must be a permutation of the 32 targets")
    for pair in pairs:
        if pair["target_id"] == pair["source_id"]:
            raise ValueError("a question cannot donate to itself")
        if str(pair["target_answer"]) == str(pair["source_answer"]):
            raise ValueError("donor and target must have different gold answers")
    counts = pd.DataFrame(pairs).groupby(list(strata)).size()
    expected = target_n // len(counts)
    if not (counts == expected).all():
        raise ValueError(f"unbalanced target strata: {counts.to_dict()}")


def _load_checkpoint(path):
    done = {}
    if not path or not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "target_id" in row:
                done[str(row["target_id"])] = row
    return done


def _append_checkpoint(path, record):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _skip_record(pair, reason):
    return {
        "example_id": pair["target_id"],
        "target_id": pair["target_id"],
        "source_id": pair["source_id"],
        "skipped": True,
        "skip_reason": reason,
        "metrics": {},
        "patching_results": {
            "token_level": [],
            "category_level": {},
            "final_multi_head": {"num_heads_patched": 0, "selected_heads": []},
        },
    }


def _joint_patch_hooks(layer_to_specs):
    def make_hook(specs):
        def hook_fn(value, hook):
            patched = value.clone()
            last_pos = patched.shape[1] - 1
            for spec in specs:
                patched[:, last_pos, spec["head"], :] = spec["vec"].to(
                    patched.device, patched.dtype
                )
            return patched

        return hook_fn

    return [
        (f"blocks.{layer}.attn.hook_z", make_hook(specs))
        for layer, specs in layer_to_specs.items()
    ]


def _merge_across_pos(bank, largest):
    better = (lambda new, old: new > old) if largest else (lambda new, old: new < old)
    merged = {}
    for label, entries in bank.items():
        for info in entries:
            key = (info["layer"], info["head"])
            if key not in merged or better(info["score"], merged[key]["score"]):
                merged[key] = {**info, "label": label}
    return merged


def run_cross_patching_dual_metric(
    df,
    pairs,
    model,
    task,
    template,
    output_paths,
    checkpoint_path=None,
    heads_per_pos=3,
    ctx=2048,
    max_generation_steps=1024,
    decoding=None,
):
    """Run MRR and JSD cross patching from one shared head sweep per token."""
    # Keep pair planning and validation usable on CPU-only machines that do not
    # install TransformerLens.  The heavy dependencies are needed only here.
    import nltk
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    from src.metrics import (
        make_jsd_metric,
        make_margin_recovery_ratio_metric,
        margin_recovery_ratio,
    )
    from src.patching import patch_attn_head_out_last_pos
    from src.patching_pipelines import patching_pipeline
    from src.tasks import AnswerTriggerNotFound
    from src.utils import (
        _merge_top_heads_for_pos_jsd,
        _merge_top_heads_for_pos_margin_ratio,
        generate_full_answer_and_get_logits,
    )

    metrics = {
        "margin": {
            "factory": make_margin_recovery_ratio_metric,
            "merge": _merge_top_heads_for_pos_margin_ratio,
            "largest": True,
            "score_key": "recovery_score",
        },
        "jsd": {
            "factory": make_jsd_metric,
            "merge": _merge_top_heads_for_pos_jsd,
            "largest": False,
            "score_key": "final_jsd_score",
        },
    }
    if heads_per_pos != 3:
        raise ValueError("the main cross experiment is fixed at k=3")
    validate_cross_pairs(pairs, target_n=32)
    device = next(model.parameters()).device
    rows = {str(row[task.id_column]): row for _, row in df.iterrows()}
    missing_ids = sorted(
        {p[k] for p in pairs for k in ("source_id", "target_id")}.difference(rows)
    )
    if missing_ids:
        raise ValueError(f"pair IDs absent from dataframe: {missing_ids}")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    exports = {name: [] for name in metrics}
    done = _load_checkpoint(checkpoint_path)

    for pair in tqdm(pairs, desc="Cross patching SVAMP/Qwen (k=3)"):
        target_id = str(pair["target_id"])
        if target_id in done:
            cached = done[target_id]
            cached_source = str(
                cached.get("source_id")
                or cached.get("margin", {}).get("source_id")
                or cached.get("jsd", {}).get("source_id")
                or ""
            )
            if cached_source and cached_source != str(pair["source_id"]):
                raise ValueError(
                    f"checkpoint donor mismatch for {target_id}: "
                    f"{cached_source} != {pair['source_id']}"
                )
            for name in metrics:
                exports[name].append(cached[name])
            continue

        source = rows[str(pair["source_id"])]
        target = rows[target_id]
        source_cot_prompt = source[template.cot_col]
        target_cot_prompt = target[template.cot_col]
        target_nocot_prompt = target[template.nocot_col] + template.corrupt_suffix

        try:
            source_text, _, source_trace = generate_full_answer_and_get_logits(
                model,
                source_cot_prompt,
                max_new_tokens=max_generation_steps,
                task=task,
                decoding=decoding,
                return_generation_trace=True,
            )
            target_text, target_cot_logits = generate_full_answer_and_get_logits(
                model,
                target_cot_prompt,
                max_new_tokens=max_generation_steps,
                task=task,
                decoding=decoding,
            )
        except AnswerTriggerNotFound as exc:
            skipped = {name: _skip_record(pair, str(exc)) for name in metrics}
            for name in metrics:
                exports[name].append(skipped[name])
            _append_checkpoint(
                checkpoint_path, {"target_id": target_id, **skipped}
            )
            continue

        corrupted_tokens = model.to_tokens(target_nocot_prompt)[:, -ctx:].to(device)
        with torch.no_grad():
            target_nocot_logits = model(corrupted_tokens)[0, -1, :]

        t_true = int(target_cot_logits.argmax(dim=-1).item())
        bound = {
            name: cfg["factory"](target_nocot_logits)
            for name, cfg in metrics.items()
        }
        baseline_jsd = bound["jsd"](target_cot_logits)(target_nocot_logits).item()
        source_prefix = model.to_tokens(source_cot_prompt, prepend_bos=True).to(device)
        banks = {name: {} for name in metrics}
        pos_hms = {name: defaultdict(list) for name in metrics}
        token_level = {name: [] for name in metrics}

        for step, generated_token in enumerate(source_trace):
            heatmaps = patching_pipeline(
                model,
                source_prefix,
                corrupted_tokens,
                metric=bound,
                patching_function=patch_attn_head_out_last_pos,
                clean_reference_logits=target_cot_logits,
            )
            token_id = int(source_prefix[0, -1].item())
            token_str = model.to_string(source_prefix[:, -1:])[0].strip()
            pos_label = nltk.pos_tag([token_str if token_str else "."])[0][1]

            with torch.no_grad():
                _, source_cache = model.run_with_cache(source_prefix[:, -ctx:])

            for name, cfg in metrics.items():
                hm = heatmaps[name]
                hm_cpu = hm.detach().cpu()
                pos_hms[name][pos_label].append(hm_cpu)
                banks[name][pos_label] = cfg["merge"](
                    banks[name].get(pos_label, []),
                    hm,
                    source_cache,
                    model.cfg.n_heads,
                    heads_per_pos,
                )
                flat = hm.flatten()
                k = min(heads_per_pos, flat.numel())
                vals, idxs = torch.topk(flat, k, largest=cfg["largest"])
                token_level[name].append(
                    {
                        "step": step,
                        "token_str": token_str,
                        "token_id": token_id,
                        "pos_tag": pos_label,
                        "top_heads": [
                            {
                                "layer": divmod(int(idx.item()), model.cfg.n_heads)[0],
                                "head": divmod(int(idx.item()), model.cfg.n_heads)[1],
                                "score": float(val.item()),
                            }
                            for val, idx in zip(vals, idxs)
                        ],
                        "hm_matrix": hm_cpu.tolist(),
                    }
                )
            del source_cache
            next_token = torch.tensor(
                [[generated_token]], dtype=source_prefix.dtype, device=device
            )
            source_prefix = torch.cat([source_prefix, next_token], dim=1)

        for name, cfg in metrics.items():
            merged = _merge_across_pos(banks[name], cfg["largest"])
            category_level = {}
            for label, entries in banks[name].items():
                category_level[label] = {
                    "top_heads": [
                        {
                            "layer": item["layer"],
                            "head": item["head"],
                            "score": float(item["score"]),
                        }
                        for item in entries
                    ],
                    "aggregated_hm_matrix": torch.mean(
                        torch.stack(pos_hms[name][label]), dim=0
                    ).tolist(),
                }

            by_layer = defaultdict(list)
            for (layer, head), info in merged.items():
                by_layer[layer].append({"head": head, "vec": info["vec"]})
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=_joint_patch_hooks(by_layer),
                    return_type="logits",
                )[0, -1, :]

            if name == "margin":
                score = margin_recovery_ratio(
                    target_cot_logits, target_nocot_logits, patched_logits
                ).item()
            else:
                score = bound["jsd"](target_cot_logits)(patched_logits).item()

            target_prob = F.softmax(target_cot_logits, dim=-1)[t_true].item()
            nocot_prob = F.softmax(target_nocot_logits, dim=-1)[t_true].item()
            patched_prob = F.softmax(patched_logits, dim=-1)[t_true].item()
            selected = [
                {
                    "layer": layer,
                    "head": head,
                    "pos_label": info["label"],
                    "patch_score": float(info["score"]),
                }
                for (layer, head), info in merged.items()
            ]
            exports[name].append(
                {
                    "example_id": target_id,
                    "target_id": target_id,
                    "source_id": str(pair["source_id"]),
                    "target_answer": pair["target_answer"],
                    "source_answer": pair["source_answer"],
                    "skipped": False,
                    "skip_reason": None,
                    "t_true_token_id": t_true,
                    "source_generation_steps": len(source_trace),
                    "source_generated_text": source_text,
                    "target_generated_text": target_text,
                    "metrics": {
                        "target_nocot_logit": target_nocot_logits[t_true].item(),
                        "target_cot_logit": target_cot_logits[t_true].item(),
                        "patched_logit": patched_logits[t_true].item(),
                        "target_nocot_prob": nocot_prob,
                        "target_cot_prob": target_prob,
                        "patched_prob": patched_prob,
                        "prob_increase_over_target_nocot": patched_prob - nocot_prob,
                        "target_cot_nocot_jsd": baseline_jsd,
                        cfg["score_key"]: score,
                    },
                    "patching_results": {
                        "token_level": token_level[name],
                        "category_level": category_level,
                        "final_multi_head": {
                            "num_heads_patched": len(selected),
                            "selected_heads": selected,
                        },
                    },
                }
            )

        checkpoint_row = {
            "target_id": target_id,
            **{name: exports[name][-1] for name in metrics},
        }
        _append_checkpoint(checkpoint_path, checkpoint_row)
        tqdm.write(
            f"{pair['source_id']} -> {target_id} | "
            f"MRR={exports['margin'][-1]['metrics']['recovery_score']:.4f} | "
            f"JSD={exports['jsd'][-1]['metrics']['final_jsd_score']:.4f}"
        )

    for name, path in output_paths.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(exports[name], handle, indent=2, ensure_ascii=False)
    return exports
