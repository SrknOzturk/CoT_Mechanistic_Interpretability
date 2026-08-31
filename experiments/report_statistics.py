"""
experiments/report_statistics.py

Produces the statistical uncertainty / sensitivity tables requested by the AC,
entirely offline from results/*.json. No GPU, no torch.

Usage:
    python experiments/report_statistics.py [--results-dir results] [--n-boot 10000]
"""

import argparse
import json
import os
import sys

import numpy as np

# insert(0), not append: an unrelated `src` package exists in site-packages and
# shadows this project's `src` if the repo root is appended instead of prepended.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis import (  # noqa: E402
    align,
    bootstrap_ci,
    fmt_ci,
    fmt_p,
    head_frequency,
    holm_bonferroni,
    k_sweep,
    load_results,
    skip_summary,
    metric_vector,
    paired_comparison,
    select_heads_for_k,
)

METRICS = [
    ("final_jsd_score", "Final JSD (lower = better)"),
    ("logit_increase", "Logit increase"),
    ("prob_increase", "Probability increase"),
]


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rdir = args.results_dir or os.path.join(root, "results")

    normal = load_results(os.path.join(rdir, "multi_head_patching_with_jsd_results.json"))
    cross = load_results(os.path.join(rdir, "multi_head_cross_patching_with_jsd_results.json"))
    rand = load_results(os.path.join(rdir, "random_patching_jsd_results.json"))

    if not normal:
        print("No normal-patching JSD results found; nothing to do.")
        return

    print(f"loaded: normal={len(normal)}  cross={len(cross)}  random={len(rand)}")

    for label, results in [("normal", normal), ("cross", cross), ("random", rand)]:
        if not results:
            continue
        s = skip_summary(results)
        if s["skipped"]:
            print(f"  [{label}] {s['skipped']}/{s['total']} examples skipped "
                  f"(no answer trigger reached):")
            for eid, reason in s["reasons"].items():
                print(f"      {eid}: {reason}")

    # -----------------------------------------------------------------
    # 0. Replication check: does the offline selector reproduce k=3?
    # -----------------------------------------------------------------
    rule("0. REPLICATION CHECK (offline head selection vs stored selection, k=3)")
    ok = bad = 0
    diffs = []
    for rec in normal:
        stored = rec["patching_results"]["final_multi_head"]["num_heads_patched"]
        recomputed = len(select_heads_for_k(rec, 3))
        if stored == recomputed:
            ok += 1
        else:
            bad += 1
            diffs.append((rec["example_id"], stored, recomputed))
    print(f"  exact match: {ok}/{ok + bad} examples")
    if diffs:
        print("  mismatches (example_id, stored, recomputed):")
        for d in diffs[:10]:
            print("   ", d)
        print("  -> the offline k-sweep cannot be trusted until this is resolved")
    else:
        print("  -> offline re-analysis reproduces the stored selection exactly")

    # -----------------------------------------------------------------
    # 1. Bootstrap CIs
    # -----------------------------------------------------------------
    rule(f"1. BOOTSTRAP CONFIDENCE INTERVALS (BCa, B={args.n_boot}, 95%)")
    for label, results in [("normal", normal), ("cross", cross), ("random", rand)]:
        if not results:
            continue
        print(f"\n  [{label} patching]  n={len(results)}")
        for key, desc in METRICS:
            ids, vals = metric_vector(results, key)
            if vals.size == 0:
                continue
            ci = bootstrap_ci(vals, n_boot=args.n_boot, seed=args.seed)
            print(f"    {desc:32s} {fmt_ci(ci)}   (sd={ci['sd']:.4f})")

    ids, base = metric_vector(rand, "baseline_jsd") if rand else ([], np.array([]))
    if base.size:
        ci = bootstrap_ci(base, n_boot=args.n_boot, seed=args.seed)
        print(f"\n  [unpatched baseline]  No-CoT vs CoT divergence")
        print(f"    {'Baseline JSD':32s} {fmt_ci(ci)}   (sd={ci['sd']:.4f})")

    # -----------------------------------------------------------------
    # 2. Paired comparisons
    # -----------------------------------------------------------------
    rule("2. PAIRED COMPARISONS (same examples, Wilcoxon signed-rank)")
    comparisons = []
    if rand:
        comparisons.append(("normal vs random", normal, rand, "final_jsd_score"))
        comparisons.append(("normal vs unpatched baseline", normal, rand, None))
    if cross:
        comparisons.append(("normal vs cross", normal, cross, "final_jsd_score"))
        comparisons.append(("cross vs random", cross, rand, "final_jsd_score"))

    rows = []
    for name, a, b, key in comparisons:
        if key is None:
            # special case: compare normal's final JSD against the stored baseline
            bmap = {str(r["example_id"]): r["metrics"].get("baseline_jsd") for r in b}
            pairs = [(r["metrics"]["final_jsd_score"], bmap[str(r["example_id"])])
                     for r in a if bmap.get(str(r["example_id"])) is not None]
            va = np.asarray([p[0] for p in pairs])
            vb = np.asarray([p[1] for p in pairs])
        else:
            _, va, vb = align(a, b, key)
        if va.size == 0:
            continue
        res = paired_comparison(va, vb, n_boot=args.n_boot, seed=args.seed)
        res["name"] = name
        rows.append(res)

    adj = holm_bonferroni([r["p_value"] for r in rows]) if rows else []
    for r, p_adj in zip(rows, adj):
        print(f"\n  {r['name']}   (n={r['n']})")
        print(f"    mean A = {r['mean_a']:.4f}   mean B = {r['mean_b']:.4f}")
        print(f"    difference (A-B) = {r['mean_diff']:+.4f}  "
              f"95% CI [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]")
        print(f"    Wilcoxon W = {r['wilcoxon_W']:.1f}   "
              f"p = {fmt_p(r['p_value'])}   p_holm = {fmt_p(float(p_adj))}")
        print(f"    rank-biserial r = {r['rank_biserial']:+.3f}   "
              f"examples where A has lower JSD: {r['n_favoring_a']}/{r['n']}")

    # -----------------------------------------------------------------
    # 3. k-sweep
    # -----------------------------------------------------------------
    rule("3. HEAD-SELECTION SENSITIVITY (k-sweep, offline)")
    print("  Note: recovery-vs-k needs one forward pass per (example, k) on GPU.")
    print("  What is computable offline is the selected head SET and its stability.\n")
    print(f"    {'k':>3s}  {'mean heads':>11s}  {'min':>4s}  {'max':>4s}  {'Jaccard vs k=3':>15s}")
    for row in k_sweep(normal, ks=(1, 3, 5, 10), reference_k=3):
        print(f"    {row['k']:3d}  {row['mean_heads']:11.2f}  {row['min_heads']:4d}  "
              f"{row['max_heads']:4d}  {row['mean_jaccard_vs_ref']:15.3f}")

    # -----------------------------------------------------------------
    # 4. Head frequency
    # -----------------------------------------------------------------
    rule("4. MOST FREQUENTLY SELECTED HEADS (across examples, k=3)")
    freq = head_frequency(normal)
    n_ex = len(normal)
    print(f"    {'layer.head':>11s}  {'count':>6s}  {'share':>7s}")
    for (layer, head), count in freq[:15]:
        print(f"    {layer:5d}.{head:<5d}  {count:6d}  {count / n_ex:6.1%}")
    print(f"\n    distinct heads ever selected: {len(freq)}")

    rule("DONE")


if __name__ == "__main__":
    main()
