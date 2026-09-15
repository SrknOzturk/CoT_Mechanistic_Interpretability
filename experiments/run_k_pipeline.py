"""Resume-safe normal -> k(1..30) -> chosen-k random -> ablation workflow.

Examples:
  python experiments/run_k_pipeline.py prepare --model olmo2-1b --dataset svamp
  python experiments/run_k_pipeline.py complete --model olmo2-1b --dataset svamp --margin-k 6 --jsd-k 15 --workers 1
"""

import argparse
import os
import subprocess
import sys
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def common(args):
    return ["--model", args.model, "--dataset", args.dataset,
            "--template", args.template, "--device", args.device]


def prepare(args):
    stem = f"{args.model}__{args.dataset}__{args.template}__normal"
    normal = [os.path.join(args.results_dir, f"{stem}__{m}.json")
              for m in ("margin", "jsd")]
    def completed(path):
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                return sum(not row.get("skipped") for row in json.load(f))
        except (OSError, json.JSONDecodeError):
            return 0

    if not all(completed(x) >= args.target_n for x in normal):
        cmd = ["experiments/run_parallel.py", *common(args), "--experiments", "normal",
               "--target-n", str(args.target_n), "--ctx", str(args.ctx),
               "--max-steps", str(args.max_steps), "--out-dir", args.results_dir,
               "--no-ablation"]
        if args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        if args.gpu_total_gb is not None:
            cmd += ["--gpu-total-gb", str(args.gpu_total_gb)]
        run(cmd)
    else:
        print("Normal patching outputs already exist; reusing them.")
    run(["experiments/rescore_k_sweep.py", *common(args), "--ctx", str(args.ctx),
         "--results-dir", args.results_dir, "--k", *map(str, range(1, 31))])


def complete(args):
    if args.margin_k is None or args.jsd_k is None:
        raise SystemExit("complete requires --margin-k and --jsd-k")
    kdir = os.path.join(args.results_dir, "k_sweep")
    stem = f"{args.model}__{args.dataset}__{args.template}"
    normal_paths = {
        "margin": os.path.join(kdir, f"{stem}__normal__margin__k{args.margin_k}.json"),
        "jsd": os.path.join(kdir, f"{stem}__normal__jsd__k{args.jsd_k}.json"),
    }
    missing = [x for x in normal_paths.values() if not os.path.exists(x)]
    if missing:
        raise SystemExit("Run the prepare stage first; missing: " + ", ".join(missing))

    run(["experiments/run_random_for_k.py", *common(args),
         "--margin-k", str(args.margin_k), "--jsd-k", str(args.jsd_k),
         "--ctx", str(args.ctx), "--seed", str(args.seed),
         "--results-dir", args.results_dir])

    for source, paths in (
        ("normal", normal_paths),
        ("random", {
            "margin": os.path.join(kdir, f"{stem}__random__margin__k{args.margin_k}.json"),
            "jsd": os.path.join(kdir, f"{stem}__random__jsd__k{args.jsd_k}.json"),
        }),
    ):
        for metric, path in paths.items():
            k = args.margin_k if metric == "margin" else args.jsd_k
            out_stem = f"{stem}__{source}__{metric}__k{k}"
            cmd = ["experiments/run_ablation_parallel.py", *common(args),
                   "--metric", metric, "--results-file", path,
                   "--output-stem", out_stem, "--results-dir", args.results_dir]
            if args.workers is not None:
                cmd += ["--workers", str(args.workers)]
            if args.gpu_total_gb is not None:
                cmd += ["--gpu-total-gb", str(args.gpu_total_gb)]
            if args.equation_ablation:
                cmd += ["--equation-ablation"]
            run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "complete", "all"])
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--dataset", default="svamp")
    ap.add_argument("--template", default="step_by_step")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--margin-k", type=int)
    ap.add_argument("--jsd-k", type=int)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--target-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--gpu-total-gb", type=float)
    ap.add_argument(
        "--equation-ablation",
        action="store_true",
        help="also run Direct-Equation ablation for SVAMP",
    )
    args = ap.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("complete", "all"):
        complete(args)


if __name__ == "__main__":
    main()
