"""
src/analysis.py

Offline statistical analysis of patching results. Pure CPU, no torch required:
everything here is a re-analysis of the matrices already stored in results/*.json.

Provides:
  * tolerant result loading (with a regex fallback for truncated JSON)
  * BCa bootstrap confidence intervals
  * paired comparisons (Wilcoxon signed-rank, rank-biserial, paired bootstrap)
  * offline k-sweep over head selection, replicating the JSD merge logic
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# ===========================================================================
# Loading
# ===========================================================================

def skip_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How many examples were skipped, and why."""
    skipped = [r for r in results if r.get("skipped")]
    return {
        "total": len(results),
        "usable": len(results) - len(skipped),
        "skipped": len(skipped),
        "reasons": {str(r["example_id"]): r.get("skip_reason") for r in skipped},
    }


def load_results(path: str, quiet: bool = False) -> List[Dict[str, Any]]:
    """
    Loads a patching results JSON. Falls back to a regex scrape if the file was
    truncated mid-write (which happened to the two margin result files).
    Returns [] if the file is missing.
    """
    if not os.path.exists(path):
        if not quiet:
            print(f"[analysis] missing: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if not quiet:
            print(f"[analysis] {os.path.basename(path)} is truncated ({e}); "
                  f"recovering per-example metrics via regex fallback")
        return _regex_recover(path)


def _regex_recover(path: str) -> List[Dict[str, Any]]:
    """Salvages example_id + metrics from a truncated results file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    out = []
    for block in content.split('"example_id":')[1:]:
        m = re.search(r'\s*"([^"]+)"', block)
        if not m:
            continue
        rec: Dict[str, Any] = {"example_id": m.group(1), "metrics": {}}
        mblock = block.split('"metrics":')
        if len(mblock) > 1:
            head = mblock[1].split("}")[0]
            for key, val in re.findall(r'"(\w+)":\s*(-?[\d.eE+]+)', head):
                rec["metrics"][key] = float(val)
        nh = re.search(r'"num_heads_patched":\s*(\d+)', block)
        if nh:
            rec["num_heads_patched"] = int(nh.group(1))
        out.append(rec)
    return out


def metric_vector(results: Sequence[Dict[str, Any]], key: str) -> Tuple[List[str], np.ndarray]:
    """
    Extracts (example_ids, values) for one metric, preserving file order.

    Examples recorded as skipped carry an empty `metrics` dict, so they drop out
    here and never reach a statistic.
    """
    ids, vals = [], []
    for r in results:
        if r.get("skipped"):
            continue
        v = r.get("metrics", {}).get(key)
        if v is None:
            continue
        ids.append(str(r["example_id"]))
        vals.append(float(v))
    return ids, np.asarray(vals, dtype=float)


def align(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]],
          key: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Aligns two result sets on example_id and returns the common ids with their
    paired metric values. Order follows `a`.

    An example skipped in either set is dropped from both, which is what keeps
    the comparison paired.
    """
    bmap = {str(r["example_id"]): r for r in b if not r.get("skipped")}
    ids, va, vb = [], [], []
    for r in a:
        if r.get("skipped"):
            continue
        eid = str(r["example_id"])
        if eid not in bmap:
            continue
        x = r.get("metrics", {}).get(key)
        y = bmap[eid].get("metrics", {}).get(key)
        if x is None or y is None:
            continue
        ids.append(eid)
        va.append(float(x))
        vb.append(float(y))
    return ids, np.asarray(va), np.asarray(vb)


# ===========================================================================
# Bootstrap
# ===========================================================================

def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bias-corrected and accelerated (BCa) bootstrap CI over examples.

    Falls back to the percentile interval when the acceleration term is
    undefined (all jackknife replicates identical).
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    observed = float(statistic(values))
    if n < 2:
        return {"mean": observed, "lo": observed, "hi": observed, "n": n, "n_boot": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.asarray([statistic(values[i]) for i in idx], dtype=float)

    # bias correction
    prop = float(np.mean(boot < observed))
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = stats.norm.ppf(prop)

    # acceleration via jackknife
    jack = np.asarray([statistic(np.delete(values, i)) for i in range(n)], dtype=float)
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    a = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    lo_q = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    hi_q = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))

    if not (np.isfinite(lo_q) and np.isfinite(hi_q)):
        lo_q, hi_q = alpha / 2, 1 - alpha / 2

    return {
        "mean": observed,
        "lo": float(np.quantile(boot, lo_q)),
        "hi": float(np.quantile(boot, hi_q)),
        "sd": float(values.std(ddof=1)),
        "n": n,
        "n_boot": n_boot,
    }


# ===========================================================================
# Paired comparison
# ===========================================================================

def rank_biserial(diff: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-).
    The appropriate effect size for a Wilcoxon signed-rank test.
    Ranges [-1, 1]; sign follows the direction of `diff`.
    """
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    wp = float(ranks[d > 0].sum())
    wn = float(ranks[d < 0].sum())
    total = wp + wn
    return 0.0 if total == 0 else (wp - wn) / total


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compares two paired conditions measured on the same examples.
    `a` is the treatment (e.g. selected-head patching), `b` the control
    (e.g. random patching). Reports the difference a - b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    try:
        w_stat, p_two = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:  # all differences zero
        w_stat, p_two = float("nan"), 1.0

    ci = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return {
        "n": int(diff.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "diff_lo": ci["lo"],
        "diff_hi": ci["hi"],
        "wilcoxon_W": float(w_stat),
        "p_value": float(p_two),
        "rank_biserial": rank_biserial(diff),
        "n_favoring_a": int(np.sum(diff < 0)),  # JSD: lower is better
    }


def holm_bonferroni(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ===========================================================================
# Offline k-sweep over head selection
# ===========================================================================

def _merge_jsd(prev: List[Dict], hm: np.ndarray, n_heads: int, k: int) -> List[Dict]:
    """
    Offline replica of utils._merge_top_heads_for_pos_jsd (lower score = better).
    Vectors are not needed here, only (layer, head, score).
    """
    flat = hm.reshape(-1)
    kk = min(k, flat.size)
    idx = np.argpartition(flat, kk - 1)[:kk] if kk < flat.size else np.arange(flat.size)

    entries = list(prev)
    for i in idx:
        entries.append({"layer": int(i) // n_heads, "head": int(i) % n_heads,
                        "score": float(flat[i])})

    best: Dict[Tuple[int, int], Dict] = {}
    for e in entries:
        key = (e["layer"], e["head"])
        if key not in best or e["score"] < best[key]["score"]:
            best[key] = e
    return sorted(best.values(), key=lambda x: x["score"])[:k]


def select_heads_for_k(record: Dict[str, Any], k: int) -> List[Tuple[int, int]]:
    """
    Replays the per-POS-tag head selection for a given k, from the saved
    per-step heatmaps. Returns the final merged head set as (layer, head) pairs.
    """
    steps = record.get("patching_results", {}).get("token_level", [])
    if not steps:
        return []

    n_heads = len(steps[0]["hm_matrix"][0])
    bank: Dict[str, List[Dict]] = defaultdict(list)
    for step in steps:
        hm = np.asarray(step["hm_matrix"], dtype=float)
        tag = step.get("pos_tag", "UNK")
        bank[tag] = _merge_jsd(bank[tag], hm, n_heads, k)

    merged: Dict[Tuple[int, int], float] = {}
    for entries in bank.values():
        for e in entries:
            key = (e["layer"], e["head"])
            if key not in merged or e["score"] < merged[key]:
                merged[key] = e["score"]
    return sorted(merged.keys())


def k_sweep(results: Sequence[Dict[str, Any]],
            ks: Sequence[int] = (1, 3, 5, 10),
            reference_k: int = 3) -> List[Dict[str, Any]]:
    """
    For each k: how many heads get selected, and how much the selected set
    overlaps the reference-k set (Jaccard). Fully offline.
    """
    rows = []
    per_example: Dict[int, List[set]] = {k: [] for k in ks}

    for rec in results:
        for k in ks:
            per_example[k].append(set(select_heads_for_k(rec, k)))

    ref_sets = per_example[reference_k]
    for k in ks:
        sizes = np.asarray([len(s) for s in per_example[k]], dtype=float)
        jac = []
        for s, ref in zip(per_example[k], ref_sets):
            union = s | ref
            jac.append(len(s & ref) / len(union) if union else 1.0)
        rows.append({
            "k": k,
            "mean_heads": float(sizes.mean()),
            "min_heads": int(sizes.min()),
            "max_heads": int(sizes.max()),
            "mean_jaccard_vs_ref": float(np.mean(jac)),
        })
    return rows


def head_frequency(results: Sequence[Dict[str, Any]],
                   key: str = "selected_heads") -> List[Tuple[Tuple[int, int], int]]:
    """How often each (layer, head) was selected across examples, most common first."""
    counter: Dict[Tuple[int, int], int] = defaultdict(int)
    for rec in results:
        heads = rec.get("patching_results", {}).get("final_multi_head", {}).get(key, [])
        for h in heads:
            counter[(int(h["layer"]), int(h["head"]))] += 1
    return sorted(counter.items(), key=lambda kv: -kv[1])


# ===========================================================================
# Formatting
# ===========================================================================

def fmt_ci(ci: Dict[str, float], digits: int = 4) -> str:
    """'0.2089 [0.1712, 0.2481]'"""
    return f"{ci['mean']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4g}"
