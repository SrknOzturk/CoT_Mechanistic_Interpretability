"""Create fixed, balanced two-statement Web of Lies datasets.

The official task is programmatically generated.  This script enumerates its
complete two-statement problem space, then selects disjoint main and backup
sets with equal counts for each underlying truth-pattern combination.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/processed"
SELECTION_SEED = 20260918
NAMES = (
    "Crista", "Lorine", "Alejandro", "Millie", "Leda", "Amberly", "Audrie",
    "Andree", "Antwan", "Michaela", "Willian", "Delfina", "Maybelle", "Jaymie",
    "Helene", "Shaunda", "Ka", "Bernita", "Conception", "Gwenn", "Alexis",
    "Christie", "Fidel", "Raymond", "Fletcher", "Kandi", "Vina", "Elanor",
    "Jerry", "Sherrie", "Sima", "Phoebe", "Teressa", "Delbert", "Inga",
    "Shenna", "Sal", "Michael", "Tamika", "Millicent", "Vernell", "Kristian",
    "Jim", "Rashida", "Yoland", "Jamey", "Ryan", "Dallas", "Osvaldo", "Shalonda",
)
PATTERNS = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)


def pattern_name(first_truth: bool, second_truth: bool) -> str:
    return f"first_{'true' if first_truth else 'false'}__second_{'true' if second_truth else 'false'}"


def statement(subject: str, truth: bool) -> str:
    return f"{subject} {'tells the truth' if truth else 'lies'}."


def all_two_statement_examples() -> dict[str, list[dict]]:
    groups = {pattern_name(*pattern): [] for pattern in PATTERNS}
    for first_name in NAMES:
        for second_name in NAMES:
            if first_name == second_name:
                continue
            for first_truth, second_truth in PATTERNS:
                first = statement(first_name, first_truth)
                second_claim = "tells the truth" if second_truth else "lies"
                question = f"{first} {second_name} says {first_name} {second_claim}."
                question += f" Does {second_name} tell the truth?"
                # A speaker is truthful precisely when their statement matches
                # the known truth value of the previous speaker.
                answer = first_truth == second_truth
                groups[pattern_name(first_truth, second_truth)].append(
                    {
                        "input": question,
                        "gold_answer": "Yes" if answer else "No",
                        "chain_length": 2,
                        "first_statement_truth": first_truth,
                        "second_statement_truth": second_truth,
                        "truth_pattern": pattern_name(first_truth, second_truth),
                    }
                )
    return groups


def choose_disjoint_sets(groups: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SELECTION_SEED)
    main, backup = [], []
    for pattern in sorted(groups):
        candidates = list(groups[pattern])
        rng.shuffle(candidates)
        main.extend(candidates[:16])
        backup.extend(candidates[16:24])
    rng.shuffle(main)
    rng.shuffle(backup)
    return main, backup


def dataset_payload(name: str, examples: list[dict], per_pattern: int) -> dict:
    return {
        "dataset": name,
        "source": "BIG-bench Web of Lies official two-statement generator",
        "selection_seed": SELECTION_SEED,
        "chain_length": 2,
        "selection": {
            "per_truth_pattern": per_pattern,
            "truth_patterns": [pattern_name(*pattern) for pattern in PATTERNS],
            "disjoint_from": "the other fixed Web of Lies length-2 dataset",
        },
        "summary": {
            "n": len(examples),
            "gold_distribution": dict(Counter(example["gold_answer"] for example in examples)),
            "truth_pattern_distribution": dict(Counter(example["truth_pattern"] for example in examples)),
        },
        "examples": [
            {"id": f"{name}_{index:03d}", **example}
            for index, example in enumerate(examples, start=1)
        ],
    }


def write_dataset(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    groups = all_two_statement_examples()
    main_examples, backup_examples = choose_disjoint_sets(groups)
    main_payload = dataset_payload("bigbench_web_of_lies_length2_balanced64", main_examples, 16)
    backup_payload = dataset_payload("bigbench_web_of_lies_length2_backup32", backup_examples, 8)

    main_ids = {example["input"] for example in main_examples}
    backup_ids = {example["input"] for example in backup_examples}
    if len(main_ids) != 64 or len(backup_ids) != 32 or main_ids & backup_ids:
        raise RuntimeError("Expected 64/32 unique, disjoint selected questions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataset(OUTPUT_DIR / "bigbench_web_of_lies_length2_balanced64.json", main_payload)
    write_dataset(OUTPUT_DIR / "bigbench_web_of_lies_length2_backup32.json", backup_payload)
    print(json.dumps({"main": main_payload["summary"], "backup": backup_payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
