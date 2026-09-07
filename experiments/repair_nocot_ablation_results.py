"""Repair No-CoT skip flags produced before the generated-only fix.

No model run is needed: all three generations and extracted answers are
already present in the CSV.  The script updates No-CoT CSVs and their summary
rows in place while leaving CoT rows untouched.
"""

import argparse
import glob
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def accuracy(df, column):
    if not len(df):
        return 0.0
    values = df[column]
    if values.dtype != bool:
        values = values.fillna(False).astype(str).str.lower().eq("true")
    return 100.0 * values.mean()


def true_count(series):
    if series.dtype == bool:
        return int(series.sum())
    return int(series.fillna(False).astype(str).str.lower().eq("true").sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    ap.add_argument("--model", default=None, help="optional filename prefix filter")
    args = ap.parse_args()
    abdir = os.path.join(args.results_dir, "ablation")
    prefix = f"{args.model}__" if args.model else "*"
    paths = sorted(glob.glob(os.path.join(abdir, f"{prefix}*__ablation_NoCoT.csv")))
    if not paths:
        raise SystemExit("No matching No-CoT ablation CSV files found")

    for path in paths:
        df = pd.read_csv(path)
        old = true_count(df["skipped"])
        # Numeric tasks produce NaN when no answer was extracted; boolean tasks
        # retain a proper True/False value. Both are handled by notna().
        df["skipped"] = ~df["normal_extracted"].notna()
        df.to_csv(path, index=False)
        usable = df[~df["skipped"]]

        stem = os.path.basename(path).removesuffix("__ablation_NoCoT.csv")
        summary_path = os.path.join(abdir, f"{stem}__ablation_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            mask = summary["condition"] == "NoCoT"
            summary.loc[mask, "normal_acc"] = accuracy(usable, "normal_correct")
            summary.loc[mask, "ablated_acc"] = accuracy(usable, "ablation_correct")
            summary.loc[mask, "random_acc"] = accuracy(usable, "random_correct")
            summary.loc[mask, "n"] = len(usable)
            summary.loc[mask, "skipped"] = len(df) - len(usable)
            summary.to_csv(summary_path, index=False)

        print(f"{os.path.basename(path)}: skipped {old} -> {int(df.skipped.sum())}; "
              f"normal={accuracy(usable, 'normal_correct'):.2f}% "
              f"ablated={accuracy(usable, 'ablation_correct'):.2f}% "
              f"random={accuracy(usable, 'random_correct'):.2f}%")


if __name__ == "__main__":
    main()
