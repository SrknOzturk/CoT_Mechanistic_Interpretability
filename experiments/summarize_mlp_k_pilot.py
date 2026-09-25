"""
Compares the k values of the MLP k-pilot side by side (run_queue.py mlp-k-pilot).

For each k: how many MLP layers the POS-guided selection picks per example,
which layers recur, the joint-patch score, and zero-ablation accuracy with the
selected layers versus an equally sized random set of layers (exact McNemar on
the paired examples).

Accuracies are re-scored from the stored generations: the answer is the first
number on the first line after the answer trigger (CoT) or of the generation
itself (No-CoT / Direct-Equation, whose prompt already ends at the trigger), and
a CoT generation that never reached the trigger has no answer. The stored
*_correct columns take the last number instead, which reads the next line's
number when the model runs on after answering.

Usage:
    python experiments/summarize_mlp_k_pilot.py
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRIGGER = "The answer is "
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
CONDITIONS = ("CoT", "NoCoT", "DirectEquation")


def answer(text, condition):
    if not isinstance(text, str):
        return None
    if condition == "CoT":
        if TRIGGER not in text:
            return None
        text = text.rsplit(TRIGGER, 1)[1]
    m = NUM.search(text.lstrip(" ").split("\n", 1)[0])
    return float(m.group().replace(",", "")) if m else None


def correct(texts, golds, condition):
    return np.array([a is not None and abs(a - float(g)) < 1e-4
                     for a, g in ((answer(t, condition), g) for t, g in zip(texts, golds))])


def mcnemar(a, b):
    b10, b01 = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if b10 + b01 == 0 else binomtest(b10, b10 + b01, 0.5).pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "results", "pilot_mlp_k"))
    ap.add_argument("--stem", default="qwen2.5-0.5b__svamp__step_by_step__mlp__normal")
    args = ap.parse_args()

    ks = sorted(int(d[1:]) for d in os.listdir(args.root) if re.fullmatch(r"k\d+", d))
    for metric, score_key in (("jsd", "final_jsd_score"), ("margin", "recovery_score")):
        print(f"\n{'=' * 100}\nselection by {metric.upper()}"
              f"  ({'lower' if metric == 'jsd' else 'higher'} patch score = closer to CoT)\n{'=' * 100}")
        print(f"{'k':>2} | {'layers/ex':>12} | {'patch':>6} | "
              + " | ".join(f"{c:^27}" for c in CONDITIONS))
        print(f"{'':>2} | {'mean (range)':>12} | {'score':>6} | "
              + " | ".join(f"{'normal sel rand   p':^27}" for _ in CONDITIONS))
        freq_lines = []
        for k in ks:
            out = os.path.join(args.root, f"k{k}")
            pj = os.path.join(out, f"{args.stem}__{metric}.json")
            if not os.path.exists(pj):
                print(f"{k:>2} | missing {os.path.basename(pj)}")
                continue
            recs = [r for r in json.load(open(pj, encoding="utf-8")) if not r.get("skipped")]
            layers = [[h["layer"] for h in r["patching_results"]["final_multi_head"]["selected_heads"]]
                      for r in recs]
            n = np.array([len(x) for x in layers])
            score = np.mean([r["metrics"][score_key] for r in recs])
            cells = []
            for cond in CONDITIONS:
                csv = os.path.join(out, "ablation", f"{args.stem}__{metric}__ablation_{cond}.csv")
                if not os.path.exists(csv):
                    cells.append(f"{'-':^27}")
                    continue
                d = pd.read_csv(csv)
                d = d[~d["skipped"].astype(bool)]
                nor = correct(d["normal_text"], d["true_answer"], cond)
                sel = correct(d["ablation_text"], d["true_answer"], cond)
                rnd = correct(d["random_text"], d["true_answer"], cond)
                cells.append(f"{100*nor.mean():5.1f} {100*sel.mean():5.1f} {100*rnd.mean():5.1f} "
                             f"{mcnemar(sel, rnd):6.3f}")
            print(f"{k:>2} | {n.mean():5.1f} ({n.min()}-{n.max():>2}) | {score:6.3f} | " + " | ".join(cells))
            c = Counter(l for x in layers for l in x)
            freq_lines.append(f"  k={k}: {len(c)} distinct layers; most selected "
                              + ", ".join(f"L{l} ({v}/{len(recs)})" for l, v in c.most_common(6)))
        print("\n".join(["", "layer recurrence (examples in which each layer is selected):"] + freq_lines))


if __name__ == "__main__":
    main()
