"""
Runs a queue of run_parallel.py jobs one after another from a single command.

A job that fails is retried, and if it still fails the queue records it and
moves on rather than stopping, so one bad job cannot cost the rest of a
multi-day run. Every job's status is written to a status file as it finishes;
re-running the same command skips completed jobs and resumes the rest (each
job also resumes internally from its own per-example checkpoints).

Presets:
  mlp-k-pilot  Qwen2.5-0.5B on a held-out SVAMP subset, MLP component, normal
               patching + ablation for each k in --ks. Each k writes to its own
               directory so no k can reuse another's checkpoints.
  mlp-full     every model x dataset with the MLP component at one --k:
               normal + random patching, each followed by its ablation.

Usage:
  python experiments/run_queue.py mlp-k-pilot
  python experiments/run_queue.py mlp-full --k 3
  python experiments/run_queue.py mlp-full --k 3 --models llama3.2-1b --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEUE_DIR = os.path.join(REPO_ROOT, "results", "queue")

MODELS = ["qwen2.5-0.5b", "olmo2-1b", "llama3.2-1b"]
DATASETS = ["svamp", "bigbench_boolean_expressions", "bigbench_web_of_lies"]

# Worker counts that survived CoT ablation on the 48 GB card: Llama's long CoT
# generations ran out of memory at 4-5 workers and completed at 3. The patching
# stage is lighter, but both stages share one --workers value.
WORKERS = {"qwen2.5-0.5b": 6, "olmo2-1b": 3, "llama3.2-1b": 3}


def pilot_jobs(args):
    jobs = []
    for k in args.ks:
        jobs.append((f"pilot_qwen_svamp_mlp_k{k}", [
            "--model", "qwen2.5-0.5b", "--dataset", "svamp",
            "--data-file", args.pilot_file, "--target-n", str(args.pilot_n),
            "--component", "mlp", "--experiments", "normal",
            "--heads-per-pos", str(k), "--equation-ablation",
            "--out-dir", os.path.join("results", "pilot_mlp_k", f"k{k}"),
            "--workers", str(args.workers or WORKERS["qwen2.5-0.5b"]),
        ]))
    return jobs


def full_jobs(args):
    if args.k is None:
        raise SystemExit("mlp-full needs --k (choose it from the mlp-k-pilot results)")
    jobs = []
    for model in args.models:
        for dataset in args.datasets:
            cmd = ["--model", model, "--dataset", dataset, "--component", "mlp",
                   "--heads-per-pos", str(args.k),
                   "--workers", str(args.workers or WORKERS[model])]
            if dataset == "svamp":
                cmd.append("--equation-ablation")
            jobs.append((f"{model}__{dataset}__mlp_k{args.k}", cmd))
    return jobs


def run_job(name, cmd_args, log_path):
    """Runs one job, streaming its output to the console and to its log."""
    cmd = [sys.executable, "-u", os.path.join("experiments", "run_parallel.py"), *cmd_args]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        # flush every line: a buffered log looks empty for hours under tail -f
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=["mlp-k-pilot", "mlp-full"])
    ap.add_argument("--k", type=int, default=None, help="units per POS for mlp-full")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="k values for mlp-k-pilot (default 1 2 3 4 5)")
    ap.add_argument("--pilot-file", default="svamp_pilot_heldout.json")
    ap.add_argument("--pilot-n", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--workers", type=int, default=None,
                    help="override the per-model worker counts")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra attempts for a failed job before moving on (default 1)")
    ap.add_argument("--rerun", action="store_true",
                    help="run jobs the status file already marks as done")
    ap.add_argument("--dry-run", action="store_true", help="print the queue and exit")
    args = ap.parse_args()

    jobs = pilot_jobs(args) if args.preset == "mlp-k-pilot" else full_jobs(args)
    tag = args.preset if args.preset == "mlp-k-pilot" else f"{args.preset}_k{args.k}"
    os.makedirs(os.path.join(QUEUE_DIR, "logs"), exist_ok=True)
    status_path = os.path.join(QUEUE_DIR, f"{tag}.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}

    print(f"queue '{tag}': {len(jobs)} job(s); status -> {status_path}")
    for i, (name, cmd) in enumerate(jobs, 1):
        state = status.get(name, {}).get("state", "pending")
        print(f"  {i:2d}. [{state:7s}] {name}: run_parallel.py {' '.join(cmd)}")
    if args.dry_run:
        return

    for name, cmd in jobs:
        if status.get(name, {}).get("state") == "done" and not args.rerun:
            print(f"\n--- {name}: already done, skipping")
            continue
        log_path = os.path.join(QUEUE_DIR, "logs", f"{name}.log")
        for attempt in range(1, args.retries + 2):
            print(f"\n{'=' * 70}\n--- {name}  (attempt {attempt})  log: {log_path}\n{'=' * 70}")
            started = time.time()
            try:
                code = run_job(name, cmd, log_path)
            except Exception as exc:  # the queue itself must never die on one job
                print(f"[queue] {name} could not start: {exc}")
                code = -1
            hours = (time.time() - started) / 3600
            status[name] = {"state": "done" if code == 0 else "failed", "exit_code": code,
                            "attempts": attempt, "hours": round(hours, 2), "log": log_path,
                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
            if code == 0:
                break
            print(f"[queue] {name} exited with {code} after {hours:.2f} h")

    print(f"\n{'=' * 70}\nqueue '{tag}' finished")
    failed = [n for n, _ in jobs if status.get(n, {}).get("state") != "done"]
    for name, _ in jobs:
        s = status.get(name, {})
        print(f"  {s.get('state', 'pending'):7s} {s.get('hours', 0):6.2f} h  {name}")
    if failed:
        print(f"\n{len(failed)} job(s) failed; see their logs, then re-run this same command "
              f"to retry only those.")
        sys.exit(1)


if __name__ == "__main__":
    main()
