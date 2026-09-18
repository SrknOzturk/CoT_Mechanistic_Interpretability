"""Create a fixed, balanced shared subset of BIG-bench Boolean Expressions."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.run_bigbench_boolean_expressions import build_problems
from src.data_loader import BIGBENCH_BOOLEAN_DEMO_QUESTIONS


LENGTHS = (4, 5, 6)
PER_LABEL = 10
SEED = 42
OUTPUT = "data/processed/bigbench_boolean_expressions_l4_l5_l6_balanced_60.json"
RESERVE_OUTPUT = "data/processed/bigbench_boolean_expressions_l5_l6_reserve_30.json"
RESERVE_COUNTS = {
    (5, "True"): 8,
    (5, "False"): 7,
    (6, "True"): 7,
    (6, "False"): 8,
}


def _record(row, length, label, ordinal, possible, reserve=False):
    prefix = "bigbench_boolean_reserve" if reserve else "bigbench_boolean"
    return {
        "id": f"{prefix}_l{length}_{label.lower()}_{ordinal:02d}",
        "source": "BIG-bench boolean_expressions",
        "length": length,
        "input": row["input"],
        "target": label,
        "seed": SEED,
        "possible_expressions_at_length": possible,
    }


def main():
    records = []
    demo_questions = {" ".join(q.split()) for q in BIGBENCH_BOOLEAN_DEMO_QUESTIONS}
    pools = {}
    primary_questions = set()
    for length in LENGTHS:
        # Asking for more examples than exist returns the full, deterministically
        # shuffled expression set. Select each label independently from it.
        rows, possible, _ = build_problems(length, 10**9, 0, SEED)
        pools[length] = (rows, possible)
        for label in ("True", "False"):
            chosen = [
                row for row in rows
                if row["gold_answer"] == label
                and " ".join(row["input"].split()) not in demo_questions
            ][:PER_LABEL]
            if len(chosen) != PER_LABEL:
                raise RuntimeError(f"Length {length} has too few {label} expressions.")
            for ordinal, row in enumerate(chosen, start=1):
                records.append(_record(row, length, label, ordinal, possible))
                primary_questions.add(" ".join(row["input"].split()))

    metadata = {
        "description": "Balanced, fixed shared subset for Qwen2.5-0.5B and OLMo2-1B experiments.",
        "source_task": "BIG-bench boolean_expressions",
        "selection_seed": SEED,
        "lengths": list(LENGTHS),
        "per_length": 20,
        "per_label_per_length": PER_LABEL,
        "total": len(records),
        "excluded_demonstrations": list(BIGBENCH_BOOLEAN_DEMO_QUESTIONS),
        "records": records,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(OUTPUT)
    for length in LENGTHS:
        subset = [r for r in records if r["length"] == length]
        print(length, len(subset), sum(r["target"] == "True" for r in subset),
              sum(r["target"] == "False" for r in subset))

    reserve_records = []
    reserve_questions = set()
    for (length, label), count in RESERVE_COUNTS.items():
        rows, possible = pools[length]
        chosen = [
            row for row in rows
            if row["gold_answer"] == label
            and " ".join(row["input"].split()) not in demo_questions
            and " ".join(row["input"].split()) not in primary_questions
            and " ".join(row["input"].split()) not in reserve_questions
        ][:count]
        if len(chosen) != count:
            raise RuntimeError(f"Length {length} has too few reserve {label} expressions.")
        for ordinal, row in enumerate(chosen, start=1):
            reserve_records.append(
                _record(row, length, label, ordinal, possible, reserve=True)
            )
            reserve_questions.add(" ".join(row["input"].split()))

    reserve_metadata = {
        "description": (
            "Thirty fixed reserve questions for replacement when a primary Boolean "
            "Expressions example cannot be patched."
        ),
        "source_task": "BIG-bench boolean_expressions",
        "selection_seed": SEED,
        "lengths": [5, 6],
        "per_length": {"5": 15, "6": 15},
        "label_totals": {"True": 15, "False": 15},
        "total": len(reserve_records),
        "excluded_demonstrations": list(BIGBENCH_BOOLEAN_DEMO_QUESTIONS),
        "excluded_primary_questions": len(primary_questions),
        "records": reserve_records,
    }
    with open(RESERVE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(reserve_metadata, f, indent=2, ensure_ascii=False)
    print(RESERVE_OUTPUT)
    for length in (5, 6):
        subset = [r for r in reserve_records if r["length"] == length]
        print(length, len(subset), sum(r["target"] == "True" for r in subset),
              sum(r["target"] == "False" for r in subset))


if __name__ == "__main__":
    main()
