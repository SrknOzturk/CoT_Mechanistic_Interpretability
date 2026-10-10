import pandas as pd

from src.cross_patching import build_balanced_cross_pairs


def _fixture_frame():
    rows = []
    for operation_count in (1, 2):
        for kind in ("Addition", "Division", "Multiplication", "Subtraction"):
            for idx in range(6):
                rows.append(
                    {
                        "ID": f"{operation_count}-{kind}-{idx}",
                        "OperationCount": operation_count,
                        "Type": kind,
                        "Answer": idx,
                    }
                )
    return pd.DataFrame(rows)


def test_balanced_cross_pairs_are_deterministic_derangement():
    frame = _fixture_frame()
    selected_a, pairs_a = build_balanced_cross_pairs(frame, target_n=32, seed=42)
    selected_b, pairs_b = build_balanced_cross_pairs(frame, target_n=32, seed=42)

    assert pairs_a == pairs_b
    assert len(selected_a) == len(selected_b) == len(pairs_a) == 32
    assert {p["target_id"] for p in pairs_a} == {p["source_id"] for p in pairs_a}
    assert all(p["target_id"] != p["source_id"] for p in pairs_a)
    assert all(p["target_answer"] != p["source_answer"] for p in pairs_a)

    counts = pd.DataFrame(pairs_a).groupby(["OperationCount", "Type"]).size()
    assert len(counts) == 8
    assert set(counts) == {4}
