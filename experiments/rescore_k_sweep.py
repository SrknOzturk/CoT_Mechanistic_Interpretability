"""Re-score saved sequential-patching heatmaps with different k values.

The expensive layer-by-head sweeps are not repeated.  For every POS category,
the saved full heatmaps are used to reproduce the original selection rule at a
new k.  The model is only used to reconstruct the selected clean activations
and run one final joint patch per (example, metric, k).

Outputs and checkpoints live below results/k_sweep, so the original run is
never modified. Re-running the same command resumes completed items.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.metrics import make_jsd_metric, margin_recovery_ratio
from src.models import MODELS
from src.tasks import TASKS, AnswerTriggerNotFound
from src.templates import DEFAULT_TEMPLATE, TEMPLATES, get_template
from src.utils import generate_full_answer_and_get_logits, generate_till_answer, load_model


METRICS = {
    "margin": {"largest": True, "score_key": "recovery_score"},
    "jsd": {"largest": False, "score_key": "final_jsd_score"},
}


def _load_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else data
    return {str(row["example_id"]): row for row in rows}


def _selection_plan(record, k, largest):
    """Return final head choices plus the clean-trace step supplying each z."""
    per_pos = defaultdict(dict)
    for token in record["patching_results"]["token_level"]:
        label, step = token["pos_tag"], int(token["step"])
        matrix = token["hm_matrix"]
        for layer, row in enumerate(matrix):
            for head, raw_score in enumerate(row):
                score = float(raw_score)
                key = (layer, head)
                old = per_pos[label].get(key)
                if old is None or ((score > old["score"]) if largest else (score < old["score"])):
                    per_pos[label][key] = {
                        "layer": layer, "head": head, "score": score,
                        "step": step, "label": label,
                    }

    category = {}
    for label, candidates in per_pos.items():
        ordered = sorted(candidates.values(), key=lambda x: x["score"], reverse=largest)
        category[label] = ordered[:k]

    merged = {}
    for entries in category.values():
        for item in entries:
            key = (item["layer"], item["head"])
            old = merged.get(key)
            if old is None or ((item["score"] > old["score"]) if largest else
                               (item["score"] < old["score"])):
                merged[key] = item
    return list(merged.values()), category


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[(str(row["example_id"]), row["metric"], int(row["k"]))] = row
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return rows


def _reconstruct_vectors(model, prompt, task, needed, ctx, expected_tokens):
    """Regenerate the deterministic trace and retain only requested head vectors."""
    vectors = {}
    current = prompt
    max_step = max(needed, default=-1)
    for step in range(max_step + 1):
        tokens = model.to_tokens(current)[:, -ctx:].to(next(model.parameters()).device)
        last_id = int(tokens[0, -1].item())
        if step < len(expected_tokens) and last_id != int(expected_tokens[step]):
            raise RuntimeError(
                f"clean trace changed at step {step}: saved token {expected_tokens[step]}, "
                f"regenerated {last_id}"
            )
        if step in needed:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            for metric, k, layer, head in needed[step]:
                vectors[(metric, k, layer, head)] = cache["z", layer][
                    0, -1, head, :
                ].detach().cpu().clone()
            del cache
        if step < max_step:
            _, current = generate_till_answer(model, current, max_new_tokens=1, task=task)
    return vectors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="svamp", choices=sorted(TASKS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--select-only", action="store_true",
                    help="write selected heads without loading the model")
    args = ap.parse_args()
    ks = sorted(set(args.k))
    if not ks or min(ks) < 1:
        raise SystemExit("every k must be a positive integer")

    stem = f"{args.model}__{args.dataset}__{args.template}__normal"
    source = {
        metric: _load_records(os.path.join(args.results_dir, f"{stem}__{metric}.json"))
        for metric in METRICS
    }
    ids = [i for i in source["margin"] if i in source["jsd"]]
    data_path = os.path.join(REPO_ROOT, "data", "processed", f"{args.dataset}_candidates.json")
    import pandas as pd
    task, template = TASKS[args.dataset], get_template(args.template)
    df = pd.read_json(data_path)
    rows = {str(row[task.id_column]): row for _, row in df.iterrows()}

    out_dir = os.path.join(args.results_dir, "k_sweep")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, f"{stem}__k_sweep.jsonl")
    done = _read_jsonl(checkpoint)
    model = None if args.select_only else load_model(args.model, args.device)

    for example_id in tqdm(ids, desc="k sweep"):
        if example_id not in rows:
            continue
        plans = {}
        for metric, cfg in METRICS.items():
            rec = source[metric][example_id]
            if rec.get("skipped"):
                continue
            for k in ks:
                plans[(metric, k)] = _selection_plan(rec, k, cfg["largest"])[0]

        if args.select_only:
            for (metric, k), selected in plans.items():
                key = (example_id, metric, k)
                if key not in done:
                    _append(checkpoint, {
                        "example_id": example_id, "metric": metric, "k": k,
                        "select_only": True, "num_heads_patched": len(selected),
                        "selected_heads": selected,
                    })
            continue

        pending = {
            (m, k): s for (m, k), s in plans.items()
            if (
                (example_id, m, k) not in done
                or done[(example_id, m, k)].get("select_only")
                # Older k-sweep checkpoints predate the clean CoT baseline
                # fields. Re-score those entries once so mixed old/new output
                # files are never produced.
                or "clean_cot_prob" not in done[(example_id, m, k)].get("metrics", {})
            )
        }
        if not pending:
            continue
        row = rows[example_id]
        cot_prompt = row[template.cot_col]
        nocot_prompt = row[template.nocot_col] + template.corrupt_suffix
        try:
            _, clean_logits = generate_full_answer_and_get_logits(model, cot_prompt, task=task)
        except AnswerTriggerNotFound as exc:
            for metric, k in pending:
                _append(checkpoint, {"example_id": example_id, "metric": metric,
                                     "k": k, "skipped": True, "skip_reason": str(exc)})
            continue
        t_true = int(clean_logits.argmax().item())
        device = next(model.parameters()).device
        corrupted_tokens = model.to_tokens(nocot_prompt)[:, -args.ctx:].to(device)
        with torch.no_grad():
            no_cot_logits = model(corrupted_tokens)[0, -1, :]
        no_prob = float(F.softmax(no_cot_logits, dim=-1)[t_true].item())
        clean_prob = float(F.softmax(clean_logits, dim=-1)[t_true].item())
        clean_no_cot_jsd = float(
            make_jsd_metric(no_cot_logits)(clean_logits)(no_cot_logits).item()
        )

        needed = defaultdict(list)
        for (metric, k), selected in pending.items():
            for item in selected:
                needed[item["step"]].append((metric, k, item["layer"], item["head"]))
        token_rows = source["margin"][example_id]["patching_results"]["token_level"]
        expected = [x["token_id"] for x in token_rows]
        vectors = _reconstruct_vectors(model, cot_prompt, task, needed, args.ctx, expected)

        for (metric, k), selected in pending.items():
            per_layer = defaultdict(list)
            for item in selected:
                per_layer[item["layer"]].append((item["head"], vectors[
                    (metric, k, item["layer"], item["head"])]))

            def make_hook(specs):
                def hook(value, hook):
                    value = value.clone()
                    for head, vec in specs:
                        value[:, -1, head, :] = vec.to(value.device, value.dtype)
                    return value
                return hook

            hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(specs))
                     for layer, specs in per_layer.items()]
            with torch.no_grad():
                patched = model.run_with_hooks(
                    corrupted_tokens, fwd_hooks=hooks, return_type="logits")[0, -1, :]
            patched_prob = float(F.softmax(patched, dim=-1)[t_true].item())
            if metric == "margin":
                score = float(margin_recovery_ratio(clean_logits, no_cot_logits, patched).item())
            else:
                score = float(make_jsd_metric(no_cot_logits)(clean_logits)(patched).item())
            result = {
                "example_id": example_id, "metric": metric, "k": k,
                "skipped": False, "t_true_token_id": t_true,
                "num_heads_patched": len(selected), "selected_heads": selected,
                "metrics": {
                    "no_cot_prob": no_prob,
                    "clean_cot_prob": clean_prob,
                    "cot_no_cot_prob_gap": clean_prob - no_prob,
                    "clean_no_cot_jsd": clean_no_cot_jsd,
                    "patched_prob": patched_prob,
                    "prob_increase": patched_prob - no_prob,
                    "logit_increase": float(patched[t_true].item() - no_cot_logits[t_true].item()),
                    METRICS[metric]["score_key"]: score,
                },
            }
            _append(checkpoint, result)

    final = _read_jsonl(checkpoint)
    for metric in METRICS:
        for k in ks:
            rows_out = [r for (eid, m, kk), r in final.items()
                        if m == metric and kk == k and not r.get("select_only")]
            if rows_out:
                path = os.path.join(out_dir, f"{stem}__{metric}__k{k}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(rows_out, f, indent=2, ensure_ascii=False)
    print(f"Saved isolated k-sweep results/checkpoints under {out_dir}")


if __name__ == "__main__":
    main()
