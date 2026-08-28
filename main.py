"""
main.py

Shows exactly what the experiments will load -- nothing more, nothing less.

The data pipeline has three stages, and their filenames are similar on purpose
(they name the same dataset at different points), which is exactly what makes
it easy to lose track of which one matters:

    raw/        the original source files, untouched
    curated     the full dataset with prompts rendered, before any sampling
    candidates  the balanced pool the experiments actually read from  <- THIS ONE

This script reads the "candidates" file for each task via src.tasks.TASKS,
which is the same lookup experiments/run_patchings.py and run_ablations.py use.
There is no second copy of that mapping here, so this script cannot drift from
what the drivers actually load.

Usage:
    python main.py                  # both datasets, one sample row each
    python main.py --dataset svamp  # one dataset only
    python main.py --full           # print every row, not just a sample
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tasks import TASKS, get_task
from src.templates import DEFAULT_TEMPLATE, get_template

ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(ROOT, "data", "processed")

LEGACY_FILES = {
    "svamp_curated_subset.json": (
        "Not part of the current pipeline. This is the exact 32-example subset "
        "used to produce the results in `results/Old results/` for the "
        "originally submitted paper (old flat column names: PromptWithCot / "
        "PromptWithoutCot, no template suffix). Kept only so those results can "
        "still be traced back to their input; new experiments never read it."
    ),
}


def wrap(text, indent="      "):
    return textwrap.fill(text, 100, initial_indent=indent, subsequent_indent=indent)


def show_dataset(task_key: str, template_key: str, full: bool):
    task = get_task(task_key)
    template = get_template(template_key)
    path = os.path.join(PROCESSED, task.dataset_file)

    print("=" * 78)
    print(f"{task_key.upper()}   (this is what experiments load: --dataset {task_key})")
    print("=" * 78)
    print(f"  file        : data/processed/{task.dataset_file}")
    print(f"  description : {task.description}")

    if not os.path.exists(path):
        print(f"  [MISSING] run: python prepare_dataset.py")
        print()
        return False

    rows = json.load(open(path, encoding="utf-8"))
    print(f"  examples    : {len(rows)}")
    print(f"  id column   : {task.id_column}   answer column: {task.answer_column}")
    print(f"  stratified by: {', '.join(task.stratify_keys) or '(none)'}")

    if task.stratify_keys:
        from collections import Counter
        key = task.stratify_keys[-1]
        counts = Counter(r.get(key) for r in rows)
        print(f"  balance ({key}): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=str)))

    if task.answer_column in rows[0]:
        from collections import Counter
        ans_counts = Counter(r[task.answer_column] for r in rows)
        if len(ans_counts) <= 6:
            print(f"  answers     : " + ", ".join(f"{k}={v}" for k, v in ans_counts.items()))

    cot_col, nocot_col = template.cot_col, template.nocot_col
    missing_cols = [c for c in (cot_col, nocot_col) if c not in rows[0]]
    if missing_cols:
        print(f"  [ERROR] template columns missing: {missing_cols} -- re-run prepare_dataset.py")
        print()
        return False

    print(f"  template    : {template.key} ({template.description})")
    print()

    to_show = rows if full else rows[:1]
    for i, row in enumerate(to_show):
        print(f"  --- example {i + 1}/{len(to_show)}"
              + ("" if full else f" of {len(rows)}") + f"  (id={row[task.id_column]!r}) ---")
        print(f"    gold answer : {row[task.answer_column]!r}")
        print(f"    CoT prompt:")
        print(wrap(row[cot_col]))
        print(f"    No-CoT prompt (+ corrupt_suffix {template.corrupt_suffix!r}):")
        print(wrap(row[nocot_col] + template.corrupt_suffix))
        print()

    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(TASKS), default=None,
                    help="show only this dataset (default: all)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, help="which prompt structure to render")
    ap.add_argument("--full", action="store_true", help="print every row instead of one sample")
    args = ap.parse_args()

    keys = [args.dataset] if args.dataset else sorted(TASKS)
    ok = all(show_dataset(k, args.template, args.full) for k in keys)

    present_legacy = {f: note for f, note in LEGACY_FILES.items()
                      if os.path.exists(os.path.join(PROCESSED, f))}
    if present_legacy:
        print("=" * 78)
        print("OTHER FILES IN data/processed/ (not used by current experiments)")
        print("=" * 78)
        for f, note in present_legacy.items():
            print(f"  {f}")
            print(wrap(note))
            print()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
