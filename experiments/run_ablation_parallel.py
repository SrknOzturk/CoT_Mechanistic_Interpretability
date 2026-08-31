"""
experiments/run_ablation_parallel.py

Standalone, process-parallel ablation for an ALREADY-COMPLETED patching run.
Splits No-CoT and CoT ablation of the accepted examples across worker
processes exactly like run_parallel.py does for patching -- same checkpoint/
resume, same OOM-retry-with-fewer-workers, same merge machinery, reused
directly from there rather than duplicated.

Use this when patching has already run (via run_parallel.py or the plain
run_patchings.py CLI) and only ablation needs (re-)running -- e.g. after
tweaking an ablation parameter, or because --no-ablation was passed the first
time.

Usage:
    python experiments/run_ablation_parallel.py --model qwen2.5-0.5b --dataset svamp
    python experiments/run_ablation_parallel.py --model qwen2.5-0.5b --dataset svamp --metric jsd
    python experiments/run_ablation_parallel.py --model qwen2.5-0.5b --dataset svamp --dry-run
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EXPERIMENTS_DIR)

# Test-only seam, same one run_parallel.py exposes -- see its module docstring.
if os.environ.get("_RUN_PARALLEL_TEST_IMPORT"):
    import importlib as _importlib
    _importlib.import_module(os.environ["_RUN_PARALLEL_TEST_IMPORT"])

import pandas as pd  # noqa: E402

import run_patchings as rp  # noqa: E402
from run_parallel import run_ablation_parallel  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.tasks import TASKS, get_task  # noqa: E402
from src.templates import DEFAULT_TEMPLATE, TEMPLATES, get_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Process-parallel ablation for an already-completed patching run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="svamp", choices=sorted(TASKS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--metric", default="both", choices=["margin", "jsd", "both"],
                    help="which of normal's two outputs to ablate the heads of")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default=rp.RESULTS_DIR,
                    help="where normal's merged {base}__{metric}.json files live")
    ap.add_argument("--out-dir", default=None,
                    help="where ablation CSVs and checkpoints go (default: <results-dir>/ablation)")

    ap.add_argument("--workers", type=int, default=None,
                    help="worker process count; omit to size automatically from GPU memory")
    ap.add_argument("--gpu-total-gb", type=float, default=None)
    ap.add_argument("--gpu-used-gb", type=float, default=None,
                    help="VRAM already occupied by other processes; omit to auto-detect")
    ap.add_argument("--worker-cap", type=int, default=8,
                    help="upper bound on auto-sized worker count (compute saturates before VRAM does)")

    ap.add_argument("--fresh", action="store_true",
                    help="discard existing ablation checkpoints and start over")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sharding plan and exit without loading a model or spawning workers")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    task = get_task(args.dataset)
    template = get_template(args.template)

    data_path = os.path.join(REPO_ROOT, "data", "processed", rp.DATASETS[args.dataset])
    if not os.path.exists(data_path):
        print(f"[ERROR] dataset not found: {data_path}")
        sys.exit(1)

    metrics = ["margin", "jsd"] if args.metric == "both" else [args.metric]
    base = rp.run_id(args.model, args.dataset, "normal", args.template)

    summary = []
    for metric in metrics:
        curated_heads_path = os.path.join(args.results_dir, f"{base}__{metric}.json")
        print(f"\n{'=' * 20} {metric} {'=' * 20}")
        if not os.path.exists(curated_heads_path):
            print(f"  Skipping: {os.path.basename(curated_heads_path)} not found. "
                  f"Run the patching experiment ('normal') first.")
            continue
        summary.extend(run_ablation_parallel(
            args, task, template, curated_heads_path, task.id_column, data_path,
            out_dir, stem=f"{base}__{metric}"))

    if args.dry_run:
        return

    if summary:
        sdf = pd.DataFrame(summary)
        summary_path = os.path.join(out_dir, f"{base}__ablation_summary.csv")
        sdf.to_csv(summary_path, index=False)
        print(f"\nSummary written to {summary_path}")
        print(sdf.to_string(index=False))
    else:
        print("\nNo ablation runs completed (no curated head files found).")


if __name__ == "__main__":
    main()
