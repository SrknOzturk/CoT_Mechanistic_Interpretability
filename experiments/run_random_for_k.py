"""Run the random-activation control with head budgets from chosen k-sweep files.

The full random head scan is shared by Margin and JSD.  Re-running resumes from
the k-specific JSONL checkpoint and never overwrites the default-k results.
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from experiments.run_patchings import DATASETS, sequential_random_patching_dual_metric
from src.models import MODELS
from src.tasks import TASKS, get_task
from src.templates import DEFAULT_TEMPLATE, TEMPLATES, get_template
from src.utils import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="svamp", choices=sorted(TASKS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--margin-k", type=int, required=True)
    ap.add_argument("--jsd-k", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()
    if args.margin_k < 1 or args.jsd_k < 1:
        raise SystemExit("k values must be positive")

    stem = f"{args.model}__{args.dataset}__{args.template}"
    k_dir = os.path.join(args.results_dir, "k_sweep")
    refs = {
        "margin": os.path.join(k_dir, f"{stem}__normal__margin__k{args.margin_k}.json"),
        "jsd": os.path.join(k_dir, f"{stem}__normal__jsd__k{args.jsd_k}.json"),
    }
    missing = [path for path in refs.values() if not os.path.exists(path)]
    if missing:
        raise SystemExit("Missing k-sweep input(s): " + ", ".join(missing))

    # Only examples present in both selected-k files are eligible.  This also
    # makes the command work with an existing partially curated result set.
    ids = None
    import json
    for path in refs.values():
        with open(path, encoding="utf-8") as f:
            current = {str(x["example_id"]) for x in json.load(f) if not x.get("skipped")}
        ids = current if ids is None else ids & current

    task, template = get_task(args.dataset), get_template(args.template)
    data_path = os.path.join(REPO_ROOT, "data", "processed", DATASETS[args.dataset])
    df = pd.read_json(data_path)
    df = df[df[task.id_column].astype(str).isin(ids)].copy()
    if df.empty:
        raise SystemExit("No common accepted examples in the selected k files")

    tag = f"margin-k{args.margin_k}__jsd-k{args.jsd_k}"
    outputs = {
        "margin": os.path.join(k_dir, f"{stem}__random__margin__k{args.margin_k}.json"),
        "jsd": os.path.join(k_dir, f"{stem}__random__jsd__k{args.jsd_k}.json"),
    }
    checkpoint = os.path.join(k_dir, f"{stem}__random__{tag}.jsonl")
    print(f"Random control: {len(df)} examples; Margin k={args.margin_k}, JSD k={args.jsd_k}")
    model = load_model(args.model, args.device)
    sequential_random_patching_dual_metric(
        df=df, model=model, id_column=task.id_column, ctx=args.ctx,
        reference_json_paths=refs, output_paths=outputs, seed=args.seed,
        task=task, template=template, checkpoint_path=checkpoint,
    )


if __name__ == "__main__":
    main()
