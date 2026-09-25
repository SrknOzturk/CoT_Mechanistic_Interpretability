"""
Draws a small SVAMP subset that the main experiment never sees, for pilots such
as choosing k before the full MLP run.

An example is held out only if it is outside the main candidate pool AND shares
no problem body with any candidate: SVAMP builds several questions from each
body, so a sibling of a main-experiment question would leak the same story into
the pilot. Sampling is stratified like the main pool, so the pilot covers the
same operation types and counts.

The file lists `per_group` primary rows per stratum first, then one reserve row
per stratum, so run_parallel's --target-n takes the primaries and backfills
skipped examples from the reserve.

Usage:
    python experiments/make_heldout_subset.py
    python experiments/make_heldout_subset.py --per-group 2 --seed 7
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data_loader import SVAMP_STRATIFY  # noqa: E402

PROCESSED = os.path.join(REPO_ROOT, "data", "processed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-group", type=int, default=2,
                    help="primary rows per OperationCount x Type stratum (default 2 -> 16 rows)")
    ap.add_argument("--reserve-per-group", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="svamp_pilot_heldout.json")
    args = ap.parse_args()

    curated = pd.read_json(os.path.join(PROCESSED, "svamp_curated.json"))
    main_pool = pd.read_json(os.path.join(PROCESSED, "svamp_candidates.json"))

    used_ids = set(main_pool["ID"].astype(str))
    used_bodies = set(main_pool["Body"].astype(str).str.strip())
    pool = curated[~curated["ID"].astype(str).isin(used_ids)
                   & ~curated["Body"].astype(str).str.strip().isin(used_bodies)
                   & (curated["OperationCount"] > 0)]

    take = args.per_group + args.reserve_per_group
    primary, reserve = [], []
    for key, group in pool.groupby(list(SVAMP_STRATIFY)):
        if len(group) < take:
            raise SystemExit(f"stratum {key} has only {len(group)} held-out rows, need {take}")
        drawn = group.sample(n=take, random_state=args.seed)
        primary.append(drawn.iloc[:args.per_group])
        reserve.append(drawn.iloc[args.per_group:])

    out = pd.concat(primary + reserve, ignore_index=True)
    path = os.path.join(PROCESSED, args.out)
    out.to_json(path, orient="records", indent=4)

    n_primary = sum(len(p) for p in primary)
    print(f"held-out pool: {len(pool)} of {len(curated)} rows "
          f"(excluded {len(used_ids)} candidate ids and every row sharing their bodies)")
    print(f"wrote {len(out)} rows -> {path}  ({n_primary} primary + {len(out) - n_primary} reserve)")
    print(out.iloc[:n_primary].groupby(list(SVAMP_STRATIFY)).size().to_string())


if __name__ == "__main__":
    main()
