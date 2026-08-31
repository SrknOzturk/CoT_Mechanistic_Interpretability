"""
prepare_dataset.py

Builds the curated tables and the candidate pools the experiments draw from.

The pools are deliberately larger than the target of 64 examples per dataset:
the pre-filter drops any example whose CoT trace never reaches the answer
anchor, and that decision is model-dependent, so the final set is the
intersection across all three models. Over-sampling keeps n=64 reachable.

Usage:
    python prepare_dataset.py                 # everything
    python prepare_dataset.py --only prontoqa
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    PRONTOQA_STRATIFY,
    SVAMP_STRATIFY,
    create_and_save_balanced_subset,
    curate_prontoqa_and_save_json,
    curate_svamp_and_save_json,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")

SVAMP_RAW = os.path.join(RAW, "SVAMP.json")
SVAMP_CURATED = os.path.join(PROCESSED, "svamp_curated.json")
SVAMP_POOL = os.path.join(PROCESSED, "svamp_candidates.json")

PRONTOQA_RAW_DIR = os.path.join(RAW, "prontoqa")
PRONTOQA_CURATED = os.path.join(PROCESSED, "prontoqa_curated.json")
PRONTOQA_POOL = os.path.join(PROCESSED, "prontoqa_candidates.json")

# The experiments keep TARGET_N examples per dataset. The pools below are
# larger on purpose: the pre-filter drops any example whose CoT trace never
# reaches the answer anchor, and because that decision is model-dependent the
# kept set is the intersection across all three models. 1.5x leaves room for
# both without having to regenerate.
TARGET_N = 64

# SVAMP strata: 2 operation counts x 4 types = 8 groups -> 8 kept per group
SVAMP_PER_GROUP = 12
# ProntoQA strata: hop levels 2-5, 25 each -> 100 examples.
# 1-hop is generated but excluded: a single deduction step gives the model
# almost nothing to reason over, so its traces are too short to scan.
PRONTOQA_HOPS = (2, 3, 4, 5)          # 4 groups -> 16 kept per group
PRONTOQA_PER_GROUP = 24


def discover_prontoqa_files(directory):
    """Maps hop count -> file, from names like '3hop_1shot_seed42.json'."""
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, "*hop_*.json"))):
        m = re.match(r"(\d+)hop", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["svamp", "prontoqa"], default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(PROCESSED, exist_ok=True)

    if args.only in (None, "svamp"):
        print("=" * 60)
        print("SVAMP")
        print("=" * 60)
        if not os.path.exists(SVAMP_RAW):
            print(f"  [skip] raw file not found: {SVAMP_RAW}")
        else:
            curate_svamp_and_save_json(SVAMP_RAW, SVAMP_CURATED)
            create_and_save_balanced_subset(
                SVAMP_CURATED, SVAMP_POOL,
                n_samples=SVAMP_PER_GROUP,
                stratify_keys=SVAMP_STRATIFY,
                random_state=args.seed,
            )

    if args.only in (None, "prontoqa"):
        print()
        print("=" * 60)
        print("ProntoQA")
        print("=" * 60)
        hop_files = discover_prontoqa_files(PRONTOQA_RAW_DIR)
        if not hop_files:
            print(f"  [skip] no hop files in {PRONTOQA_RAW_DIR}")
            print("  generate them from prontoqa-main with:")
            print("    python run_experiment.py --model-name json --model-size dummy")
            print("      --ontology fictional --num-trials 40 --few-shot-examples 1")
            print("      --min-hops 1 --max-hops 5 --hops-skip 1 --seed 42")
        else:
            print(f"  found hops: {sorted(hop_files)}")
            curate_prontoqa_and_save_json(hop_files, PRONTOQA_CURATED)
            create_and_save_balanced_subset(
                PRONTOQA_CURATED, PRONTOQA_POOL,
                n_samples=PRONTOQA_PER_GROUP,
                stratify_keys=PRONTOQA_STRATIFY,
                random_state=args.seed,
                drop_zero_operations=False,
                keep_groups={"hop": PRONTOQA_HOPS},
            )

    print()
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
