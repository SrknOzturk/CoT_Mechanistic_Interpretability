"""
experiments/plan_compute.py

Prints how many workers fit on the GPU and what the full experiment costs.

Usage:
    python experiments/plan_compute.py
    python experiments/plan_compute.py --gpu-total 48 --gpu-used 0 --n 100
    python experiments/plan_compute.py --no-shared-sweep   # separate margin/JSD scans
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gpu_planning import (
    estimate_cell,
    estimate_worker_memory,
    max_workers,
    seconds_per_token_step,
    wall_clock_hours,
)
from src.models import MODELS, get_model_spec

# Trace lengths are guesses until the pre-filter measures them. SVAMP's is
# anchored on the published runs (mean 46 steps); the step-by-step cue tends to
# lengthen traces, and ProntoQA proofs are longer still.
DATASETS = {
    "svamp": dict(n=100, mean_steps=55, seq_len=260),
    "prontoqa": dict(n=100, mean_steps=85, seq_len=520),
}

# The experiment set is one sequential scan (scoring margin and JSD together)
# plus its two single-position random controls. Cross-patching was dropped.
SEQUENTIAL_SCANS = 1
RANDOM_SWEEPS = 2


def rule(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-total", type=float, default=48.0, help="GPU memory in GB")
    ap.add_argument("--gpu-used", type=float, default=0.0, help="GB already occupied")
    ap.add_argument("--n", type=int, default=None, help="override examples per dataset")
    ap.add_argument("--efficiency", type=float, default=0.65,
                    help="fraction of ideal speedup that concurrency actually delivers")
    ap.add_argument("--worker-cap", type=int, default=8,
                    help="stop adding workers past this, where compute saturates")
    ap.add_argument("--no-slicing", action="store_true", help="cost without last-position slicing")
    ap.add_argument("--no-shared-sweep", action="store_true",
                    help="score margin and JSD in separate scans instead of one "
                         "(what the code did before the combined scan)")
    args = ap.parse_args()

    slicing = not args.no_slicing
    # both metrics read the same patched logits, so one scan can score both
    sequential_runs = 2 if args.no_shared_sweep else 1

    rule("MEMORY PER WORKER  (float32, one model copy per process)")
    print(f"  {'model':<14}{'weights':>9}{'cache':>9}{'logits':>9}{'runtime':>9}{'total':>9}   workers")
    workers = {}
    for key, spec in MODELS.items():
        seq = max(d["seq_len"] for d in DATASETS.values())
        m = estimate_worker_memory(spec, seq_len=seq)
        w = max_workers(spec, args.gpu_total, args.gpu_used, seq_len=seq,
                        hard_cap=args.worker_cap)
        workers[key] = w
        print(f"  {key:<14}{m.weights_gb:8.1f}G{m.cache_gb:8.1f}G{m.logits_gb:8.1f}G"
              f"{m.overhead_gb:8.1f}G{m.total_gb:8.1f}G{w:10d}")
    print(f"  free VRAM assumed: {args.gpu_total - args.gpu_used:.0f} GB"
          f"   (worker cap {args.worker_cap}, set by compute saturation not memory)")

    rule("COST PER TOKEN STEP  (one full sweep over every layer x head)")
    print(f"  {'model':<14}{'heads':>7}{'s/pass':>10}{'s/step':>10}")
    for key, spec in MODELS.items():
        print(f"  {key:<14}{spec.n_head_slots:7d}"
              f"{seconds_per_token_step(spec, slicing) / spec.n_head_slots:9.3f}s"
              f"{seconds_per_token_step(spec, slicing):9.1f}s")
    if slicing:
        print("  (includes the 22% saved by slicing the residual to the last position)")

    rule(f"GPU HOURS  (n={args.n or 'per-dataset default'} examples, "
         f"{'separate' if args.no_shared_sweep else 'shared'} margin/JSD scan)")
    print(f"  {'model':<14}{'dataset':<11}{'n':>5}{'steps':>7}{'scans':>7}{'GPU-h':>9}")
    total = 0.0
    rows = []
    for key, spec in MODELS.items():
        for ds, cfg in DATASETS.items():
            n = args.n or cfg["n"]
            runs = sequential_runs * SEQUENTIAL_SCANS
            c = estimate_cell(spec, ds, n, cfg["mean_steps"],
                              sequential_runs=runs, sweep_runs=RANDOM_SWEEPS, slicing=slicing)
            rows.append((key, ds, runs, c.gpu_hours))
            total += c.gpu_hours
            print(f"  {key:<14}{ds:<11}{n:5d}{cfg['mean_steps']:7d}{runs:7d}"
                  f"{c.gpu_hours:8.1f}h")
    print(f"  {'':<14}{'':<11}{'':>5}{'':>7}{'TOTAL':>7}{total:8.1f}h")

    rule("WALL CLOCK  (models run one at a time; examples spread across workers)")
    print(f"  {'model':<14}{'workers':>9}{'GPU-h':>9}{'wall-h':>9}{'days':>8}")
    grand_wall = 0.0
    for key in MODELS:
        gpu_h = sum(h for k, _, _, h in rows if k == key)
        wall = wall_clock_hours(gpu_h, workers[key], args.efficiency)
        grand_wall += wall
        print(f"  {key:<14}{workers[key]:9d}{gpu_h:8.1f}h{wall:8.1f}h{wall / 24:8.1f}")
    print(f"  {'TOTAL':<14}{'':>9}{total:8.1f}h{grand_wall:8.1f}h{grand_wall / 24:8.1f}")
    print()
    print(f"  assumes {args.efficiency:.0%} scaling efficiency; measure it with a"
          f" calibration sweep before trusting it")


if __name__ == "__main__":
    main()
