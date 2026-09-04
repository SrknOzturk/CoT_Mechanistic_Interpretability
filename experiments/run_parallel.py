"""
experiments/run_parallel.py

Splits one model's candidate examples across several OS processes, each with
its own CUDA context and its own copy of the model in VRAM, then merges their
checkpoints into the same output files experiments/run_patchings.py produces.

Models still run one at a time -- Qwen finishes before OLMo starts, which
finishes before Llama starts, since a worker pool is sized per model. What
runs concurrently is examples of the SAME model.

Why not batch examples into one process instead: unequal prompt lengths would
need left-padding, and the patch site is defined as the literal last position
-- padding would move it. N independent single-example processes sidestep
that, at the cost of one CUDA context and one copy of the model's weights per
worker (which is why this only makes sense for models small enough that
several copies fit in VRAM at once).

Every worker checkpoints per example (the checkpoint_path support added to
run_patchings.py's three long-running functions), so re-running the exact
same command after a crash, an OOM, or a Ctrl-C continues from wherever it
stopped -- nothing already written is recomputed. Pass --fresh to discard
existing checkpoints for this (model, dataset, template, experiment) and
start over.

If a worker hits CUDA OOM, its shard's progress is unaffected (each shard
writes to its own file, never touched by another worker) and the whole
experiment is retried with one fewer worker, re-sharding only what is still
pending. It gives up only once a single worker alone cannot fit.

Usage:
    python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp
    python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp --workers 6
    python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp \\
        --gpu-total-gb 24 --dry-run
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EXPERIMENTS_DIR)

# Test-only seam: importing a module for its side effects lets a test install
# stand-ins for transformer_lens/nltk/etc. before run_patchings is imported below.
# This crosses process boundaries too -- multiprocessing's spawn start method
# re-runs this file's top level in every worker, so the env var (inherited from
# the parent) is what makes the substitution happen there as well. Production
# runs never set this, so it is a no-op.
if os.environ.get("_RUN_PARALLEL_TEST_IMPORT"):
    import importlib as _importlib
    _importlib.import_module(os.environ["_RUN_PARALLEL_TEST_IMPORT"])

import pandas as pd  # noqa: E402

import run_patchings as rp  # noqa: E402
from plan_compute import DATASETS as COST_TABLE  # noqa: E402  (seq_len per dataset)
from src.gpu_planning import max_workers as gpu_max_workers  # noqa: E402
from src.models import MODELS, get_model_spec  # noqa: E402
from src.tasks import TASKS  # noqa: E402
from src.templates import DEFAULT_TEMPLATE, TEMPLATES  # noqa: E402

# A worker that dies from CUDA OOM exits with this code; anything else is a
# real bug and stops the whole run rather than being silently retried.
OOM_EXIT_CODE = 42


def _resolve_loader(rp_module):
    """
    load_model, or a test-only stand-in.

    The seam is read wherever a model actually gets loaded -- inside a worker
    process and in the orchestrator's own ablation stage alike -- so a test
    can substitute a scripted model in both places with one env var.
    Production runs never set this.
    """
    loader = rp_module.load_model
    override = os.environ.get("_RUN_PARALLEL_TEST_LOADER")
    if override:
        import importlib
        mod_name, attr = override.split(":")
        loader = getattr(importlib.import_module(mod_name), attr)
    return loader


# ===========================================================================
# Worker process
# ===========================================================================

def _worker_main(cfg):
    """
    Runs in a freshly spawned process. Loads its own model, restricts the
    candidate dataframe to its shard, and calls the same experiment function
    experiments/run_patchings.py uses -- with output writing disabled and
    checkpointing turned on, so this process's only externally visible effect
    is its own checkpoint file.
    """
    import sys as _sys
    _sys.path.insert(0, cfg["repo_root"])
    _sys.path.insert(0, cfg["experiments_dir"])

    import pandas as _pd
    import run_patchings as _rp
    from src.tasks import get_task as _get_task
    from src.templates import get_template as _get_template

    try:
        loader = _resolve_loader(_rp)

        # test-only seam: force exactly one simulated OOM per marker file, to
        # exercise the retry-with-fewer-workers path without a real GPU.
        oom_marker = os.environ.get("_RUN_PARALLEL_TEST_OOM_MARKER")
        if oom_marker and not os.path.exists(oom_marker):
            open(oom_marker, "w").close()
            raise RuntimeError("CUDA out of memory (simulated for testing)")

        model = loader(cfg["model"], device=cfg["device"])

        full_df = _pd.read_json(cfg["data_path"])
        id_col = cfg["id_column"]
        shard_ids = set(cfg["shard_ids"])
        shard_df = full_df[full_df[id_col].astype(str).isin(shard_ids)].copy()
        order = {sid: i for i, sid in enumerate(cfg["shard_ids"])}
        shard_df["_order"] = shard_df[id_col].astype(str).map(order)
        shard_df = shard_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

        task = _get_task(cfg["dataset"])
        template = _get_template(cfg["template"])
        name = cfg["experiment"]

        kwargs = dict(
            df=shard_df, model=model, id_column=id_col, task=task, template=template,
            ctx=cfg["ctx"], seed=cfg["seed"], checkpoint_path=cfg["checkpoint_path"],
        )
        if name in _rp.MULTI_OUTPUT:
            kwargs["output_paths"] = {m: None for m in _rp.MULTI_OUTPUT[name]}
            kwargs["heads_per_pos"] = cfg["heads_per_pos"]
            if name in _rp.RANDOM_REFERENCE_MULTI:
                # a dict of {"margin": path, "jsd": path}, unlike the plain
                # string the legacy single-metric random controls take below
                kwargs["reference_json_paths"] = cfg["reference_json_path"]
            else:
                kwargs["max_generation_steps"] = cfg["max_steps"]
        else:
            kwargs["output_json_path"] = None
            key = "jsd_heads_per_pos" if name.endswith("jsd") else "margin_ratio_heads_per_pos"
            kwargs[key] = cfg["heads_per_pos"]
            if name in _rp.RANDOM_REFERENCE:
                kwargs["reference_json_path"] = cfg["reference_json_path"]
            else:
                kwargs["max_generation_steps"] = cfg["max_steps"]

        _rp.EXPERIMENTS[name](**kwargs)
        _sys.exit(0)

    except Exception as exc:  # noqa: BLE001 -- classify and report, don't swallow
        import traceback
        msg = str(exc).lower()
        is_oom = (
            type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in msg
            or "cuda oom" in msg
        )
        traceback.print_exc()
        _sys.exit(OOM_EXIT_CODE if is_oom else 1)


# ===========================================================================
# Orchestration
# ===========================================================================

def _query_gpu_free_gb(device_index=0):
    """Returns (free_gb, total_gb), or (None, None) if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info(device_index)
        return free / (1024 ** 3), total / (1024 ** 3)
    except Exception:
        return None, None


def _resolve_worker_count(args):
    if args.workers is not None:
        return args.workers

    spec = get_model_spec(args.model)
    total, used = args.gpu_total_gb, args.gpu_used_gb
    if total is None:
        free, detected_total = _query_gpu_free_gb()
        if free is None:
            raise SystemExit(
                "Could not query GPU memory automatically (no CUDA visible here). "
                "Pass --gpu-total-gb (and optionally --gpu-used-gb) explicitly, "
                "or --workers to skip auto-sizing entirely."
            )
        total = detected_total
        if used is None:
            used = total - free
    used = used or 0.0

    seq_len = COST_TABLE.get(args.dataset, {}).get("seq_len", 400)
    n = gpu_max_workers(spec, total, used, seq_len=seq_len, hard_cap=args.worker_cap)
    print(f"[auto] {args.model}: {total:.0f} GB total, {used:.1f} GB already used "
          f"-> {n} worker(s) (cap {args.worker_cap})")
    return n


def _read_checkpoints(ckpt_glob):
    """Every checkpoint line ever written for a pattern, keyed by example_id."""
    by_id = {}
    for path in sorted(glob.glob(ckpt_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    by_id[str(rec["example_id"])] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return by_id


def _pending_ids(all_ids, ckpt_glob):
    done = _read_checkpoints(ckpt_glob)
    return [i for i in all_ids if str(i) not in done]


def _is_skipped(record):
    """
    True if a checkpoint record represents a skipped example.

    Handles both shapes checkpoints take: the flat one random_margin/
    random_jsd write (skipped is a top-level key, absent entirely on a
    success), and the one the dual-metric ("normal") scan writes, which
    nests one sub-record per metric -- {"example_id":..., "margin": {...},
    "jsd": {...}} -- so the flag has to be read from inside those.
    """
    if "skipped" in record:
        return bool(record["skipped"])
    for value in record.values():
        if isinstance(value, dict) and "skipped" in value:
            return bool(value["skipped"])
    return False


def _shard(ids, n_shards):
    """Splits into up to n_shards contiguous, roughly-equal, non-empty chunks."""
    if not ids:
        return []
    n_shards = max(1, min(n_shards, len(ids)))
    q, r = divmod(len(ids), n_shards)
    shards, start = [], 0
    for i in range(n_shards):
        size = q + (1 if i < r else 0)
        shards.append(ids[start:start + size])
        start += size
    return [s for s in shards if s]


def _merge(ckpt_dir, base, out_dir, metrics=None, keep_ids=None):
    """
    Reads every checkpoint line ever written for this experiment across all
    attempts and workers, keeps the last write per example_id (defensive; the
    sharding design never lets the same id appear twice within one attempt),
    and writes the final results file(s) at the paths run_patchings.py and
    run_ablations.py expect.

    keep_ids, when given, restricts the output to exactly those ids -- used to
    trim away skipped examples and any unused reserve draws so the final file
    holds exactly the target count of successfully-patched examples.
    """
    pattern = os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")
    by_id = _read_checkpoints(pattern)
    if keep_ids is not None:
        by_id = {k: v for k, v in by_id.items() if k in keep_ids}

    if metrics:
        out = {m: [] for m in metrics}
        for rec in by_id.values():
            for m in metrics:
                out[m].append(rec[m])
        for m, records in out.items():
            path = os.path.join(out_dir, f"{base}__{m}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
            print(f"  wrote {path}  ({len(records)} records)")
        return out

    records = list(by_id.values())
    path = os.path.join(out_dir, base + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    print(f"  wrote {path}  ({len(records)} records)")
    return records


def _ablation_worker_main(cfg):
    """
    Runs in a freshly spawned process. Loads its own model, restricts the
    candidate dataframe to its shard, reloads curated_heads fresh from
    cfg["curated_heads_path"] (a completed patching run's merged results
    file), and runs one ablation condition (No-CoT or CoT) over the shard.

    Same exit-code contract as _worker_main: 0 success, OOM_EXIT_CODE on a
    CUDA OOM, 1 on anything else.
    """
    import sys as _sys
    _sys.path.insert(0, cfg["repo_root"])
    _sys.path.insert(0, cfg["experiments_dir"])

    import pandas as _pd
    import run_patchings as _rp
    from src.ablation import (
        load_heads_from_experiment as _load_heads,
        run_cot_ablation_using_curated_heads as _run_cot,
        run_nocot_ablation_using_curated_heads as _run_nocot,
    )
    from src.tasks import get_task as _get_task
    from src.templates import get_template as _get_template

    try:
        loader = _resolve_loader(_rp)
        model = loader(cfg["model"], device=cfg["device"])

        full_df = _pd.read_json(cfg["data_path"])
        id_col = cfg["id_column"]
        shard_ids = set(cfg["shard_ids"])
        shard_df = full_df[full_df[id_col].astype(str).isin(shard_ids)].copy()
        order = {sid: i for i, sid in enumerate(cfg["shard_ids"])}
        shard_df["_order"] = shard_df[id_col].astype(str).map(order)
        shard_df = shard_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

        curated_heads = _load_heads(cfg["curated_heads_path"])
        task = _get_task(cfg["dataset"])
        template = _get_template(cfg["template"])

        runner = _run_nocot if cfg["condition"] == "NoCoT" else _run_cot
        runner(shard_df, model, curated_heads, task=task, template=template,
              checkpoint_path=cfg["checkpoint_path"])
        _sys.exit(0)

    except Exception as exc:  # noqa: BLE001 -- classify and report, don't swallow
        import traceback
        msg = str(exc).lower()
        is_oom = (
            type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in msg
            or "cuda oom" in msg
        )
        traceback.print_exc()
        _sys.exit(OOM_EXIT_CODE if is_oom else 1)


def _run_ablation_workers_until_processed(label, args, ids, id_column, data_path,
                                          curated_heads_path, condition, ckpt_dir, base):
    """
    Ablation's counterpart to _run_workers_until_processed: same shard / spawn
    / OOM-retry-with-fewer-workers shape, targeting _ablation_worker_main
    instead. Kept as a separate function rather than a shared one so the
    already-verified patching path stays untouched.
    """
    pattern = os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")
    workers = _resolve_worker_count(args)
    attempt = 0

    while True:
        pending = _pending_ids(ids, pattern)
        if not pending:
            return

        shards = _shard(pending, workers)
        print(f"\n[{label}] attempt {attempt}: {len(pending)}/{len(ids)} examples "
              f"pending, {len(shards)} worker(s)")

        if args.dry_run:
            for i, s in enumerate(shards):
                print(f"    worker {i}: {len(s)} examples ({s[0]}..{s[-1]})")
            return

        ctx = mp.get_context("spawn")
        procs = []
        for i, shard_ids in enumerate(shards):
            ckpt_path = os.path.join(ckpt_dir, f"{base}.attempt{attempt}.worker{i}.jsonl")
            cfg = dict(
                repo_root=REPO_ROOT, experiments_dir=EXPERIMENTS_DIR,
                model=args.model, device=args.device, dataset=args.dataset,
                template=args.template, id_column=id_column, condition=condition,
                checkpoint_path=ckpt_path, shard_ids=shard_ids,
                data_path=data_path, curated_heads_path=curated_heads_path,
            )
            p = ctx.Process(target=_ablation_worker_main, args=(cfg,), name=f"{label}-w{i}")
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        codes = [p.exitcode for p in procs]

        n_oom = sum(1 for c in codes if c == OOM_EXIT_CODE)
        n_bad = sum(1 for c in codes if c not in (0, OOM_EXIT_CODE))
        if n_bad:
            raise RuntimeError(
                f"[{label}] {n_bad}/{len(codes)} worker(s) failed for a non-memory reason "
                f"(exit codes: {codes}). Checkpoints are untouched under {ckpt_dir} -- "
                f"fix the issue and rerun this exact command to resume."
            )
        if n_oom:
            if workers <= 1:
                raise RuntimeError(f"[{label}] out of memory even with a single worker.")
            workers -= 1
            attempt += 1
            print(f"[{label}] {n_oom} worker(s) hit OOM; retrying pending examples "
                  f"with {workers} worker(s)")
            continue

        attempt += 1
        # loop back: pending is recomputed against the checkpoints just written


def run_ablation_parallel(args, task, template, curated_heads_path, id_column, data_path,
                          out_dir, stem):
    """
    Process-parallel ablation for one curated-heads source (one metric's
    accepted examples from a patching run): shards those examples across
    workers for each of No-CoT and CoT, exactly like patching's own
    parallelism, then merges into the same CSVs run_ablations.py's sequential
    path would have produced.
    """
    from src.ablation import load_heads_from_experiment
    from src.utils import safe_accuracy

    curated_heads = load_heads_from_experiment(curated_heads_path)
    if not curated_heads:
        print(f"  no heads found in {os.path.basename(curated_heads_path)}, skipping")
        return []

    accepted_ids = [str(h["example_id"]) for h in curated_heads]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    summary = []
    for condition in ("NoCoT", "CoT"):
        base = f"{stem}__{condition}"
        if args.fresh:
            removed = 0
            for p in glob.glob(os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")):
                os.remove(p)
                removed += 1
            if removed:
                print(f"[{base}] --fresh: removed {removed} existing checkpoint file(s)")

        _run_ablation_workers_until_processed(
            f"ablation-{stem}-{condition}", args, accepted_ids, id_column, data_path,
            curated_heads_path, condition, ckpt_dir, base)
        if args.dry_run:
            continue

        pattern = os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")
        by_id = _read_checkpoints(pattern)
        records = [by_id[eid] for eid in accepted_ids if eid in by_id]

        out_csv = os.path.join(out_dir, f"{stem}__ablation_{condition}.csv")
        result_df = pd.DataFrame(records)
        result_df.to_csv(out_csv, index=False)

        usable = result_df[~result_df["skipped"]] if "skipped" in result_df.columns else result_df
        n_skipped = len(result_df) - len(usable)
        row = {
            "source": stem,
            "condition": condition,
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


def _run_workers_until_processed(name, args, ids, id_column, data_path, ckpt_dir, base, ref_path):
    """
    Shards `ids` across workers and runs them, retrying with fewer workers on
    OOM, until every id in `ids` has a checkpoint record. Does not look at
    whether those records are accepted or skipped -- that is the caller's job.
    """
    pattern = os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")
    workers = _resolve_worker_count(args)
    attempt = 0

    while True:
        pending = _pending_ids(ids, pattern)
        if not pending:
            return

        shards = _shard(pending, workers)
        print(f"\n[{name}] attempt {attempt}: {len(pending)}/{len(ids)} examples "
              f"pending, {len(shards)} worker(s)")

        if args.dry_run:
            for i, s in enumerate(shards):
                print(f"    worker {i}: {len(s)} examples ({s[0]}..{s[-1]})")
            return

        ctx = mp.get_context("spawn")
        procs = []
        for i, shard_ids in enumerate(shards):
            ckpt_path = os.path.join(ckpt_dir, f"{base}.attempt{attempt}.worker{i}.jsonl")
            cfg = dict(
                repo_root=REPO_ROOT, experiments_dir=EXPERIMENTS_DIR,
                model=args.model, device=args.device, dataset=args.dataset,
                template=args.template, experiment=name, id_column=id_column,
                ctx=args.ctx, heads_per_pos=args.heads_per_pos, max_steps=args.max_steps,
                seed=args.seed, checkpoint_path=ckpt_path, shard_ids=shard_ids,
                data_path=data_path, reference_json_path=ref_path,
            )
            p = ctx.Process(target=_worker_main, args=(cfg,), name=f"{name}-w{i}")
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        codes = [p.exitcode for p in procs]

        n_oom = sum(1 for c in codes if c == OOM_EXIT_CODE)
        n_bad = sum(1 for c in codes if c not in (0, OOM_EXIT_CODE))
        if n_bad:
            raise RuntimeError(
                f"[{name}] {n_bad}/{len(codes)} worker(s) failed for a non-memory reason "
                f"(exit codes: {codes}). Checkpoints are untouched under {ckpt_dir} -- "
                f"fix the issue and rerun this exact command to resume."
            )
        if n_oom:
            if workers <= 1:
                raise RuntimeError(f"[{name}] out of memory even with a single worker.")
            workers -= 1
            attempt += 1
            print(f"[{name}] {n_oom} worker(s) hit OOM; retrying pending examples "
                  f"with {workers} worker(s)")
            continue

        attempt += 1
        # loop back: pending is recomputed against the checkpoints just written


def run_experiment(name, args, primary_ids, reserve_ids, id_column, data_path, ref_path=None):
    """
    Processes `primary_ids` (the target count), then draws from `reserve_ids`
    one batch at a time to replace any that never reach the answer trigger --
    the same skip a base model can hit on any prompt, checked here rather than
    left to run_patchings.py alone since only the orchestrator knows there is
    a reserve to draw from. Stops as soon as `len(primary_ids)` examples are
    accepted, or the reserve runs out.

    Because acceptance depends only on (model, CoT prompt) -- identical across
    normal/random_margin/random_jsd -- the same ids end up accepted for every
    experiment of a given model, as long as they are all run with the same
    --target-n/--n against the same candidate file.
    """
    target_n = len(primary_ids)
    base = rp.run_id(args.model, args.dataset, name, args.template)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    pattern = os.path.join(ckpt_dir, f"{base}.attempt*.worker*.jsonl")

    if args.fresh:
        removed = 0
        for p in glob.glob(pattern):
            os.remove(p)
            removed += 1
        if removed:
            print(f"[{name}] --fresh: removed {removed} existing checkpoint file(s)")

    attempted = list(primary_ids)
    reserve_pool = list(reserve_ids)
    round_no = 0
    t0 = time.time()
    accepted = []

    while True:
        _run_workers_until_processed(name, args, attempted, id_column, data_path,
                                     ckpt_dir, base, ref_path)
        if args.dry_run:
            return None

        records = _read_checkpoints(pattern)
        accepted = [i for i in attempted if not _is_skipped(records[str(i)])]
        shortfall = target_n - len(accepted)
        print(f"[{name}] round {round_no}: {len(accepted)}/{target_n} accepted "
              f"({len(attempted) - len(accepted)} skipped across {len(attempted)} attempted)")

        if shortfall <= 0:
            break
        if not reserve_pool:
            print(f"[{name}] reserve exhausted -- stopping with {len(accepted)}/{target_n}")
            break

        pull = reserve_pool[:shortfall]
        reserve_pool = reserve_pool[len(pull):]
        attempted += pull
        round_no += 1
        print(f"[{name}] drawing {len(pull)} example(s) from reserve "
              f"({len(reserve_pool)} left in reserve)")

    elapsed = time.time() - t0
    print(f"[{name}] finished in {elapsed / 3600:.2f} GPU-process-hours (wall clock; "
          f"overlapping workers already counted once each)")
    print(f"[{name}] merging to exactly the {len(accepted)} accepted example(s)...")
    return _merge(ckpt_dir, base, args.out_dir, metrics=rp.MULTI_OUTPUT.get(name),
                 keep_ids=set(accepted))


def _ablation_outputs_for_experiment(name, result):
    """Return ``(metric, records)`` pairs produced by a patching experiment."""
    if isinstance(result, dict):
        return list(result.items())
    if name.endswith("_margin"):
        return [("margin", result)]
    if name.endswith("_jsd"):
        return [("jsd", result)]
    return []


def _run_ablation_stage(args, task, template, experiment, result, id_column, data_path):
    """
    Runs No-CoT and CoT ablation after a patching experiment produces accepted
    examples, once per output metric, verifying the heads that experiment
    selected -- process-parallel, the same way patching itself is.

    Each metric's curated heads are reloaded by
    run_ablation_parallel from the merged file _merge() already wrote for
    the source experiment, since that is the shape workers need to reload from
    disk anyway. This includes random-activation patching sources; their
    ablation is distinct from the random-head control computed inside every
    ablation run.
    """
    outputs = _ablation_outputs_for_experiment(experiment, result)
    if not any(records for _, records in outputs):
        print(f"\n[ablation] skipped: '{experiment}' produced no accepted examples")
        return

    print("\n" + "=" * 60)
    print(f"ABLATION  (verifying the heads '{experiment}' selected, in parallel)")
    print("=" * 60)

    ablation_out_dir = os.path.join(args.out_dir, "ablation")
    os.makedirs(ablation_out_dir, exist_ok=True)

    base = rp.run_id(args.model, args.dataset, experiment, args.template)
    summary = []
    multi_output = experiment in rp.MULTI_OUTPUT
    for metric, records in outputs:
        curated_heads_path = os.path.join(
            args.out_dir, f"{base}__{metric}.json" if multi_output else f"{base}.json")
        if not records:
            print(f"\n[ablation] {metric}: no accepted examples, skipping")
            continue
        print(f"\n--- ablating heads selected by {metric} ---")
        summary.extend(run_ablation_parallel(
            args, task, template, curated_heads_path, id_column, data_path,
            ablation_out_dir,
            stem=f"{base}__{metric}" if multi_output else base))

    if summary:
        sdf = pd.DataFrame(summary)
        summary_path = os.path.join(ablation_out_dir, f"{base}__ablation_summary.csv")
        sdf.to_csv(summary_path, index=False)
        print(f"\n[ablation] summary written to {summary_path}")
        print(sdf.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Process-parallel runner for run_patchings.py experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="qwen2.5-0.5b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="svamp", choices=sorted(TASKS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--experiments", nargs="*", default=rp.DEFAULT_ORDER,
                    choices=sorted(rp.EXPERIMENTS), metavar="EXP",
                    help=f"any of: {', '.join(sorted(rp.EXPERIMENTS))} "
                         f"(default order: {' '.join(rp.DEFAULT_ORDER)})")
    ap.add_argument("--target-n", type=int, default=64,
                    help="successfully-patched examples to collect (default: 64)")
    ap.add_argument("--n", type=int, default=None,
                    help="limit the candidate pool to the first N rows before splitting into "
                         "--target-n primary + reserve (default: use the whole file)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--heads-per-pos", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=rp.RESULTS_DIR)

    ap.add_argument("--workers", type=int, default=None,
                    help="worker process count; omit to size automatically from GPU memory")
    ap.add_argument("--gpu-total-gb", type=float, default=None)
    ap.add_argument("--gpu-used-gb", type=float, default=None,
                    help="VRAM already occupied by other processes; omit to auto-detect")
    ap.add_argument("--worker-cap", type=int, default=8,
                    help="upper bound on auto-sized worker count (compute saturates before VRAM does)")

    ap.add_argument("--no-ablation", dest="ablation", action="store_false", default=True,
                    help="skip the automatic No-CoT/CoT ablation run that otherwise follows "
                         "each completed normal or random-activation patching experiment")
    ap.add_argument("--fresh", action="store_true",
                    help="discard existing checkpoints for the requested experiments and start over")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sharding plan and exit without loading a model or spawning workers")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data_path = os.path.join(REPO_ROOT, "data", "processed", rp.DATASETS[args.dataset])
    if not os.path.exists(data_path):
        print(f"[ERROR] dataset not found: {data_path}")
        sys.exit(1)

    task = TASKS[args.dataset]
    from src.templates import get_template
    template = get_template(args.template)

    df = pd.read_json(data_path)
    if args.n is not None:
        df = df.head(args.n)
    id_column = task.id_column
    missing = [c for c in (template.cot_col, template.nocot_col, id_column) if c not in df.columns]
    if missing:
        print(f"[ERROR] missing column(s) {missing}; re-run prepare_dataset.py")
        sys.exit(1)

    all_ids = [str(x) for x in df[id_column].tolist()]
    if args.target_n > len(all_ids):
        print(f"[ERROR] --target-n {args.target_n} exceeds the candidate pool size "
              f"({len(all_ids)}); lower --target-n or widen the pool (drop --n, or "
              f"raise the per-group sample count in prepare_dataset.py)")
        sys.exit(1)
    primary_ids, reserve_ids = all_ids[:args.target_n], all_ids[args.target_n:]
    print(f"Loaded {len(all_ids)} candidates from {rp.DATASETS[args.dataset]} "
          f"({len(primary_ids)} primary + {len(reserve_ids)} reserve) "
          f"(model={args.model}, task={task.key}, template={template.key})")

    for name in args.experiments:
        if name in rp.RANDOM_REFERENCE_MULTI:
            refs, missing_refs = {}, []
            for metric, ref_spec in rp.RANDOM_REFERENCE_MULTI[name].items():
                ref_exp, ref_metric = ref_spec.split("__")
                path = os.path.join(
                    args.out_dir, rp.run_id(args.model, args.dataset, ref_exp, args.template)
                    + f"__{ref_metric}.json")
                refs[metric] = path
                if not os.path.exists(path):
                    missing_refs.append(path)
            if missing_refs:
                print(f"\nSkipping {name}: reference run(s) missing "
                      f"({', '.join(os.path.basename(p) for p in missing_refs)}). "
                      f"Run the experiment that produces them first.")
                continue
            ref_path = refs
        elif name in rp.RANDOM_REFERENCE:
            ref_exp, ref_metric = rp.RANDOM_REFERENCE[name].split("__")
            ref_path = os.path.join(
                args.out_dir, rp.run_id(args.model, args.dataset, ref_exp, args.template)
                + f"__{ref_metric}.json")
            if not os.path.exists(ref_path):
                print(f"\nSkipping {name}: reference run missing ({os.path.basename(ref_path)}). "
                      f"Run '{ref_exp}' first.")
                continue
        else:
            ref_path = None

        result = run_experiment(name, args, primary_ids, reserve_ids, id_column, data_path,
                                ref_path=ref_path)

        if args.ablation and result:
            _run_ablation_stage(
                args, task, template, name, result, id_column, data_path)

    print("\nAll requested experiments completed.")


if __name__ == "__main__":
    main()
