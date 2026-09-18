"""Official-style scoring for the programmatic BIG-bench Boolean Expressions task.

This mirrors the task's expression grammar and three direct-answer shots.
It scores the next-token probabilities of ``True`` and ``False`` rather than
asking the model to generate a long explanation.
"""

import argparse
import json
import os
import random
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models import MODELS
from src.utils import load_model


CONSTANTS = ("True", "False")
NEXT = {
    "start": ("True", "False", "not", "("),
    "True": ("or", "and", ")"),
    "False": ("or", "and", ")"),
    "not": ("True", "False", "not", "("),
    "or": ("True", "False", "not", "("),
    "and": ("True", "False", "not", "("),
    "(": ("True", "False", "not", "("),
    ")": ("or", "and", ")"),
}


def expressions_of_length(length, prefix=("start",), open_parens=0):
    if length == 0:
        tokens = prefix[1:]
        if open_parens == 0 and tokens[-1] in ("True", "False", ")"):
            yield tokens
        return
    for token in NEXT[prefix[-1]]:
        balance = open_parens + (token == "(") - (token == ")")
        if balance >= 0:
            yield from expressions_of_length(length - 1, prefix + (token,), balance)


def evaluate(tokens):
    expression = " ".join(tokens)
    return str(eval(expression)), expression + " is "


def build_problems(length, n, shots, seed):
    expressions = sorted(expressions_of_length(length))
    rng = random.Random(seed)
    rng.shuffle(expressions)
    selected = expressions[:n * (shots + 1)]
    effective_n = min(n, len(selected))
    effective_shots = len(selected) // effective_n - 1
    rows = []
    for start in range(0, effective_n * (effective_shots + 1), effective_shots + 1):
        target = selected[start]
        context = ""
        for demo in selected[start + 1:start + 1 + effective_shots]:
            answer, text = evaluate(demo)
            context += text + answer + " . "
        answer, text = evaluate(target)
        rows.append({"input": text, "gold_answer": answer, "prompt": context + text})
    return rows, len(expressions), effective_shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen2.5-0.5b")
    ap.add_argument("--length", type=int, default=7)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results/analysis")
    args = ap.parse_args()
    if not 1 <= args.length <= 14:
        raise ValueError("Length must be between 1 and 14.")

    rows, possible, effective_shots = build_problems(args.length, args.n, args.shots, args.seed)
    model = load_model(args.model)
    device = next(model.parameters()).device
    true_id = model.to_tokens("True", prepend_bos=False)[0, 0].item()
    false_id = model.to_tokens("False", prepend_bos=False)[0, 0].item()

    for row in rows:
        tokens = model.to_tokens(row["prompt"], prepend_bos=True).to(device)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        scores = torch.log_softmax(logits, dim=-1)
        row["logprob_True"] = float(scores[true_id])
        row["logprob_False"] = float(scores[false_id])
        row["prediction"] = "True" if scores[true_id] >= scores[false_id] else "False"
        row["correct"] = row["prediction"] == row["gold_answer"]

    summary = {
        "length": args.length, "possible_expressions": possible, "requested_n": args.n,
        "n": len(rows), "shots": effective_shots,
        "correct": sum(row["correct"] for row in rows),
        "accuracy_pct": round(100 * sum(row["correct"] for row in rows) / len(rows), 2),
        "predicted_True": sum(row["prediction"] == "True" for row in rows),
        "predicted_False": sum(row["prediction"] == "False" for row in rows),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.model}__bigbench_boolean_expressions__length{args.length}__{len(rows)}q"
    with open(os.path.join(args.output_dir, stem + ".jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(args.output_dir, stem + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
