"""
experiments/run_ablations.py

Zero-ablation verification experiments.

For each source of curated heads (a patching results file), ablates those heads
and measures accuracy under two prompt conditions:
    1. No-CoT ablation
    2. CoT ablation

The Direct-Equation condition was dropped from the experiment set; its pipeline
is still available in src/ablation.py for reference.

Each condition is scored three ways: unablated (normal), selected-head ablation,
and random-head ablation (the control).

Usage:
    python experiments/run_ablations.py --model qwen2.5-0.5b --dataset svamp
    python experiments/run_ablations.py --results multi_head_patching_with_jsd_results.json
"""

import argparse
import os
import sys

import pandas as pd

# insert(0), not append: an unrelated `src` package exists in site-packages and
# shadows this project's `src` when the repo root is only appended.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ablation import (
    load_heads_from_experiment,
    run_cot_ablation_using_curated_heads,
    run_nocot_ablation_using_curated_heads,
)
from src.tasks import TASKS, get_task
from src.templates import DEFAULT_TEMPLATE, TEMPLATES, get_template
from src.utils import load_model, safe_accuracy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# dataset file and answer handling both come from the TaskSpec
DATASETS = {k: t.dataset_file for k, t in TASKS.items()}

# Ablation is driven by the primary (normal sequential) patching run.
def default_results(model, dataset, template):
    """
    Primary (normal sequential) runs for this model/dataset/template.

    Matches run_id()+MULTI_OUTPUT's naming exactly: "{model}__{dataset}__
    {template}__normal__{metric}.json" -- the metric is its own "__"-separated
    suffix, not concatenated onto "normal".
    """
    stem = f'{model}__{dataset}__{template}__normal'
    return [stem + '__margin.json', stem + '__jsd.json']

CONDITIONS = [
    ("NoCoT", run_nocot_ablation_using_curated_heads),
    ("CoT", run_cot_ablation_using_curated_heads),
]


def run_ablation_conditions(model, sampled_df, task, template, curated_heads, out_dir, stem,
                            max_examples=None):
    """
    Runs NoCoT + CoT ablation for one curated-heads source (one metric's
    accepted examples from a patching run), restricted to exactly the
    examples curated_heads has head selections for.

    That restriction matters: sampled_df is the full candidate pool (primary
    + reserve), while curated_heads only covers the accepted subset a
    patching run actually selected heads for. Ablating an example with zero
    selected heads isn't a null result, it's a different experiment (nothing
    to ablate), and mixing it in dilutes the accuracy figures.
    """
    accepted_ids = {str(h["example_id"]) for h in curated_heads}
    id_col = task.id_column
    df = sampled_df[sampled_df[id_col].astype(str).isin(accepted_ids)].reset_index(drop=True)
    if len(df) < len(accepted_ids):
        print(f"  [WARNING] {len(accepted_ids) - len(df)} curated example_id(s) not found "
              f"in the candidate pool; check --dataset/--target-n match the patching run")
    print(f"  {len(df)} examples with curated heads (of {len(sampled_df)} in the candidate pool)")

    summary = []
    for name, runner in CONDITIONS:
        result_df = runner(df, model, curated_heads, max_examples=max_examples,
                           task=task, template=template)
        out_csv = os.path.join(out_dir, f"{stem}__ablation_{name}.csv")
        result_df.to_csv(out_csv, index=False)

        usable = result_df[~result_df["skipped"]] if "skipped" in result_df.columns else result_df
        n_skipped = len(result_df) - len(usable)
        row = {
            "source": stem,
            "condition": name,
            "normal_acc": safe_accuracy(usable, "normal_correct"),
            "ablated_acc": safe_accuracy(usable, "ablation_correct"),
            "random_acc": safe_accuracy(usable, "random_correct"),
            "n": len(usable),
            "skipped": n_skipped,
        }
        summary.append(row)
        print(f"      normal {row['normal_acc']:.2f}%   "
              f"ablated {row['ablated_acc']:.2f}%   "
              f"random {row['random_acc']:.2f}%   "
              f"(n={len(usable)}, {n_skipped} skipped)   -> {os.path.basename(out_csv)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--dataset", default="svamp", choices=sorted(DATASETS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--out-dir", default=None,
                    help="where the ablation CSVs go (default: <results-dir>/ablation)")
    ap.add_argument("--results", nargs="*", default=None,
                    help="patching results filenames to draw curated heads from")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    task = get_task(args.dataset)
    template = get_template(args.template)

    # validate the data before paying for a model load
    data_path = os.path.join(REPO_ROOT, "data", "processed", DATASETS[args.dataset])
    if not os.path.exists(data_path):
        print(f"[ERROR] dataset not found: {data_path}")
        sys.exit(1)
    sampled_df = pd.read_json(data_path)
    print(f"Loaded {len(sampled_df)} examples from {DATASETS[args.dataset]} "
          f"(task={task.key}, template={template.key})")

    print(f"Loading model {args.model} ...")
    model = load_model(args.model, device=args.device)

    summary = []
    for file_name in (args.results or default_results(args.model, args.dataset, args.template)):
        file_path = os.path.join(args.results_dir, file_name)
        print(f"\n{'=' * 20} {file_name} {'=' * 20}")

        curated_heads = load_heads_from_experiment(file_path)
        if not curated_heads:
            print(f"  no heads found, skipping")
            continue

        stem = os.path.splitext(file_name)[0]
        summary.extend(run_ablation_conditions(
            model, sampled_df, task, template, curated_heads, out_dir, stem,
            max_examples=args.max_examples))

    if summary:
        sdf = pd.DataFrame(summary)
        summary_path = os.path.join(
            out_dir, f"{args.model}__{args.dataset}__{args.template}__ablation_summary.csv")
        sdf.to_csv(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")
        print(sdf.to_string(index=False))
    else:
        print("\nNo ablation runs completed (no curated head files found).")


if __name__ == "__main__":
    main()
