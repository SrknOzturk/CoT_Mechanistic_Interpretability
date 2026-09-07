"""Classify saved SVAMP ablation generations without rerunning the model."""

import argparse
import ast
import csv
import json
import math
import os
import re
from collections import Counter

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
TOK_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|-?\d+(?:\.\d+)?|[^\w\s]")
EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+*xX×/÷-])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
NAME_STOP = {"The", "How", "What", "Then", "First", "So", "Total", "Let", "They", "There", "Each", "After", "Before", "If", "A"}


def numbers(text):
    return [float(x) for x in NUM_RE.findall(str(text))]


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-6)


def prompt_parts(prompt):
    pieces = str(prompt).split("Q:")
    demo = pieces[1] if len(pieces) > 2 else ""
    target = pieces[-1].split("A:", 1)[0]
    return demo, target


def repeated_ngram_fraction(tokens, n=4):
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def invalid_equations(text):
    bad = 0
    total = 0
    for a, op, b, c in EQ_RE.findall(str(text)):
        a, b, c = float(a), float(b), float(c)
        if op in ("*", "x", "X", "×"):
            expected = a * b
        elif op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif b != 0:
            expected = a / b
        else:
            continue
        total += 1
        bad += not close(expected, c)
    return int(bad), total


def classify(example_id, source, text_kind, row, text, tokenizer=None):
    text = str(text or "")
    toks = TOK_RE.findall(text)
    demo, target = prompt_parts(row["CotPrompt"])
    gold = float(row["true_answer"])
    vals = numbers(text)
    target_vals, demo_vals = numbers(target), numbers(demo)
    demo_only = {x for x in demo_vals if not any(close(x, y) for y in target_vals + [gold])}
    target_names = set(NAME_RE.findall(target)) - NAME_STOP
    demo_names = (set(NAME_RE.findall(demo)) - NAME_STOP) - target_names
    lower = text.lower()
    final_correct = str(row[f"{text_kind}_correct"]).lower() == "true"
    bad_eq, equation_count = invalid_equations(text)
    four_rep = repeated_ngram_fraction(toks, 4)
    result = {
        "example_id": example_id,
        "source": source,
        "text_kind": text_kind,
        "num_heads_ablated": int(row["num_heads_ablated"]),
        "correct": final_correct,
        "word_count": len(text.split()),
        "lexical_token_count": len(toks),
        "model_token_count": len(tokenizer.encode(text, add_special_tokens=False)) if tokenizer else None,
        "answer_marker": "the answer is" in lower,
        "gold_anywhere": any(close(x, gold) for x in vals),
        "gold_before_wrong_final": (not final_correct and any(close(x, gold) for x in vals)),
        "fourgram_repetition": four_rep,
        "loop": len(toks) >= 80 and four_rep >= 0.35,
        "equation_count": equation_count,
        "invalid_equation_count": bad_eq,
        "invalid_arithmetic": bad_eq > 0,
        "demo_number_leak": any(any(close(x, d) for d in demo_only) for x in vals),
        "demo_name_leak": any(re.search(rf"\b{re.escape(name)}\b", text) for name in demo_names),
        "target_number_coverage": (sum(any(close(x, y) for x in vals) for y in set(target_vals)) / len(set(target_vals))) if target_vals else None,
    }
    return result


def rate(series):
    return 100 * float(series.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="olmo2-1b")
    ap.add_argument("--dataset", default="svamp")
    ap.add_argument("--template", default="step_by_step")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--use-tokenizer", action="store_true")
    args = ap.parse_args()
    stem = f"{args.model}__{args.dataset}__{args.template}"
    abdir = os.path.join(args.results_dir, "ablation")
    tokenizer = None
    if args.use_tokenizer:
        from transformers import AutoTokenizer
        from src.models import MODELS
        tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model].tl_name)

    records = []
    for selection in ("normal", "random"):
        for metric in ("margin", "jsd"):
            source = f"{selection}_{metric}"
            path = os.path.join(abdir, f"{stem}__{selection}__{metric}__k{args.k}__ablation_CoT.csv")
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                records.append(classify(row["example_id"], source, "ablation", row, row["ablation_text"], tokenizer))
                records.append(classify(row["example_id"], source, "random", row, row["random_text"], tokenizer))
                if source == "normal_margin":
                    records.append(classify(row["example_id"], "baseline", "normal", row, row["normal_text"], tokenizer))

    detail = pd.DataFrame(records)
    summary_rows = []
    for (source, kind), g in detail.groupby(["source", "text_kind"], sort=False):
        summary_rows.append({
            "source": source, "text_kind": kind, "n": len(g),
            "accuracy_pct": rate(g.correct),
            "median_words": float(g.word_count.median()),
            "mean_words": float(g.word_count.mean()),
            "loop_pct": rate(g.loop),
            "answer_marker_pct": rate(g.answer_marker),
            "gold_anywhere_pct": rate(g.gold_anywhere),
            "gold_before_wrong_final_pct": rate(g.gold_before_wrong_final),
            "invalid_arithmetic_pct": rate(g.invalid_arithmetic),
            "demo_number_leak_pct": rate(g.demo_number_leak),
            "demo_name_leak_pct": rate(g.demo_name_leak),
            "mean_target_number_coverage_pct": 100 * float(g.target_number_coverage.mean()),
            "median_model_tokens": (float(g.model_token_count.median()) if tokenizer else None),
        })
    summary = pd.DataFrame(summary_rows)
    outdir = os.path.join(args.results_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    detail_path = os.path.join(outdir, f"{stem}__k{args.k}__generation_error_detail.csv")
    summary_path = os.path.join(outdir, f"{stem}__k{args.k}__generation_error_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved {detail_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
