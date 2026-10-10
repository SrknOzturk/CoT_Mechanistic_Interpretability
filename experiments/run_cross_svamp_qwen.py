"""Run fixed 32-question SVAMP/Qwen cross-patching and zero-ablation.

For every target B, a different question A supplies CoT activations.  They are
patched into B's No-CoT run and evaluated against B's own clean CoT reference.
The experiment uses k=3 per POS category and scores MRR and JSD from a shared
head sweep.  It then zero-ablates each metric's cross-selected heads under CoT
and No-CoT prompting, including unablated and equal-count random-head controls.

Examples:
    python experiments/run_cross_svamp_qwen.py --plan-only
    python experiments/run_cross_svamp_qwen.py --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cross_patching import build_balanced_cross_pairs
from src.tasks import get_task
from src.templates import get_template


MODEL = "qwen2.5-0.5b"
DATASET = "svamp"
TEMPLATE = "step_by_step"
TARGET_N = 32
K = 3


def _accuracy(frame, column):
    if frame.empty:
        return 0.0
    values = frame[column]
    if values.dtype != bool:
        values = values.fillna(False).astype(str).str.lower().eq("true")
    return 100.0 * float(values.mean())


def run_ablation_stage(
    selected,
    pairs,
    cross_exports,
    model,
    task,
    template,
    out_dir,
    stem,
    seed,
    max_steps,
):
    """Ablate cross-selected heads for both metrics and prompt conditions."""
    from src.ablation import (
        run_cot_ablation_using_curated_heads,
        run_nocot_ablation_using_curated_heads,
    )

    source_by_target = {str(p["target_id"]): str(p["source_id"]) for p in pairs}
    conditions = {
        "CoT": run_cot_ablation_using_curated_heads,
        "NoCoT": run_nocot_ablation_using_curated_heads,
    }
    summary = []

    for metric, records in cross_exports.items():
        curated = []
        for record in records:
            if record.get("skipped"):
                continue
            selected_heads = (
                record.get("patching_results", {})
                .get("final_multi_head", {})
                .get("selected_heads", [])
            )
            curated.append(
                {
                    "example_id": str(record["target_id"]),
                    "selected_heads": [
                        {"layer": int(head["layer"]), "head": int(head["head"])}
                        for head in selected_heads
                    ],
                }
            )

        accepted = {row["example_id"] for row in curated}
        ablation_df = selected[
            selected[task.id_column].astype(str).isin(accepted)
        ].reset_index(drop=True)
        print(
            f"Ablation stage ({metric}): {len(ablation_df)}/{len(selected)} "
            "targets have completed cross-selected heads"
        )
        if ablation_df.empty:
            print(f"  no usable {metric} head selections; skipping ablation")
            continue

        for condition, runner in conditions.items():
            checkpoint = out_dir / f"{stem}__{metric}__ablation_{condition}.checkpoint.jsonl"
            common = dict(
                sampled_df=ablation_df,
                model=model,
                curated_normal_results=curated,
                random_seed=seed,
                task=task,
                template=template,
                checkpoint_path=str(checkpoint),
            )
            if condition == "CoT":
                result = runner(
                    **common,
                    max_cot_reasoning_tokens=max_steps,
                )
            else:
                result = runner(
                    **common,
                    max_new_tokens=max_steps,
                )

            result.insert(
                1,
                "source_id",
                result["example_id"].astype(str).map(source_by_target),
            )
            result.insert(2, "selection_metric", metric)
            csv_path = out_dir / f"{stem}__{metric}__ablation_{condition}.csv"
            result.to_csv(csv_path, index=False)

            usable = result[~result["skipped"]] if "skipped" in result else result
            row = {
                "selection_metric": metric,
                "condition": condition,
                "normal_acc": _accuracy(usable, "normal_correct"),
                "ablated_acc": _accuracy(usable, "ablation_correct"),
                "random_acc": _accuracy(usable, "random_correct"),
                "n": len(usable),
                "skipped": len(result) - len(usable),
            }
            summary.append(row)
            print(
                f"  {condition}: normal={row['normal_acc']:.2f}% "
                f"selected={row['ablated_acc']:.2f}% random={row['random_acc']:.2f}% "
                f"(n={row['n']}, skipped={row['skipped']}) -> {csv_path.name}"
            )

    summary_frame = pd.DataFrame(summary)
    summary_path = out_dir / f"{stem}__ablation_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"Ablation summary -> {summary_path}")
    return summary_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "cross"))
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write and validate the 32 donor-target pairs without loading the model",
    )
    args = parser.parse_args()

    data_path = REPO_ROOT / "data" / "processed" / "svamp_candidates.json"
    df = pd.read_json(data_path)
    task = get_task(DATASET)
    template = get_template(TEMPLATE)
    selected, pairs = build_balanced_cross_pairs(
        df,
        target_n=TARGET_N,
        seed=args.seed,
        id_column=task.id_column,
        answer_column=task.answer_column,
        strata=task.stratify_keys,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{MODEL}__{DATASET}__{TEMPLATE}__cross__k{K}__n{TARGET_N}"
    pair_path = out_dir / f"{stem}__pairs.json"
    with open(pair_path, "w", encoding="utf-8") as handle:
        json.dump(pairs, handle, indent=2, ensure_ascii=False)

    counts = pd.DataFrame(pairs).groupby(list(task.stratify_keys)).size()
    print(f"Prepared {len(pairs)} deterministic cross pairs -> {pair_path}")
    print(counts.to_string())
    if args.plan_only:
        return

    # Import model dependencies only for the actual GPU run.  --plan-only can
    # therefore validate the experimental design in a lightweight environment.
    from src.cross_patching import run_cross_patching_dual_metric
    from src.utils import load_model

    model = load_model(MODEL, device=args.device)
    output_paths = {
        "margin": str(out_dir / f"{stem}__margin.json"),
        "jsd": str(out_dir / f"{stem}__jsd.json"),
    }
    cross_exports = run_cross_patching_dual_metric(
        selected,
        pairs,
        model,
        task=task,
        template=template,
        output_paths=output_paths,
        checkpoint_path=str(out_dir / f"{stem}.checkpoint.jsonl"),
        heads_per_pos=K,
        ctx=args.ctx,
        max_generation_steps=args.max_steps,
    )
    run_ablation_stage(
        selected,
        pairs,
        cross_exports,
        model,
        task,
        template,
        out_dir,
        stem,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
