"""
src/data_loader.py

Dataset curation. Each curator turns a raw dataset into a flat table with the
same interface, whatever the task:

    ID                      example identifier
    PromptWithoutExample    the bare question (context + query)
    Answer                  gold answer, as a string
    <stratify keys>         columns used to build a balanced subset
    PromptWithCot__<tpl>    clean side of the patching pair
    PromptWithoutCot__<tpl> corrupted side (before the answer trigger is appended)

Following the original design, the CoT demonstration is a single fixed exemplar
shared by every row, while the No-CoT demonstration is drawn per row from the
dataset itself, matched on the task's grouping key.
"""

import json
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.templates import TEMPLATES, TemplateSpec, check_template, get_template


# ===========================================================================
# Shared helpers
# ===========================================================================

def _ensure_qmark(s: str) -> str:
    s = str(s).strip()
    return s if s.endswith("?") else (s + "?")


def _collapse(s: str) -> str:
    return " ".join(str(s).split())


def _add_prompt_columns(
    df: pd.DataFrame,
    cot_demo: Dict[str, str],
    nocot_demos: List[Dict[str, str]],
    templates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Renders every registered template into its own pair of columns.

    cot_demo    : the single fixed exemplar used on the CoT side
    nocot_demos : one demo per row, used on the No-CoT side
    """
    keys = list(templates) if templates else list(TEMPLATES)
    if len(nocot_demos) != len(df):
        raise ValueError(f"need one No-CoT demo per row ({len(nocot_demos)} vs {len(df)})")

    for key in keys:
        spec: TemplateSpec = get_template(key)
        cot_col, nocot_col = [], []
        for (_, row), nocot_demo in zip(df.iterrows(), nocot_demos):
            target = {"question": row["PromptWithoutExample"]}
            cot_col.append(spec.render_cot(target, cot_demo))
            nocot_col.append(spec.render_nocot(target, nocot_demo))
        df[spec.cot_col] = cot_col
        df[spec.nocot_col] = nocot_col

        # cheap invariant guard -- catches a malformed template before any GPU time
        check_template(spec, {"question": df["PromptWithoutExample"].iloc[0]}, cot_demo)

    return df


def load_curated_data(json_path: str) -> pd.DataFrame:
    return pd.read_json(json_path)


# ===========================================================================
# SVAMP
# ===========================================================================

SVAMP_COT_DEMO = {
    "question": (
        "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
        "Each can has 3 tennis balls. How many tennis balls does he have now?"
    ),
    "reasoning": (
        "Roger started with 5 balls. Each can has 3 balls, so total balls from cans = 2 * 3 = 6. "
        "Then total = 5 + 6 = 11."
    ),
    "answer": "11",
}

SVAMP_STRATIFY = ("OperationCount", "Type")


def curate_svamp_and_save_json(raw_json_path: str, output_path: str,
                               templates: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(json.load(open(raw_json_path, "r", encoding="utf-8")))

    # one record in the public release carries a misspelled operation type
    df.loc[df["Type"] == "Common-Divison", "Type"] = "Common-Division"

    df["OperationCount"] = df["Equation"].str.count(r"[+\-*/]")

    df["PromptWithoutExample"] = (
        df["Body"].fillna("").astype(str).str.strip() + " "
        + df["Question"].fillna("").astype(str).str.strip()
    ).map(_collapse).apply(_ensure_qmark)

    # No-CoT demonstration: the next row of the same operation type, wrapping around
    demo_body = df.groupby("Type")["Body"].shift(-1).fillna(df.groupby("Type")["Body"].transform("first"))
    demo_q = df.groupby("Type")["Question"].shift(-1).fillna(df.groupby("Type")["Question"].transform("first"))
    demo_a = df.groupby("Type")["Answer"].shift(-1).fillna(df.groupby("Type")["Answer"].transform("first"))

    demo_bq = (demo_body.fillna("").astype(str).str.strip() + " "
               + demo_q.fillna("").astype(str).str.strip()).map(_collapse).apply(_ensure_qmark)

    nocot_demos = [{"question": q, "reasoning": "", "answer": str(a)}
                   for q, a in zip(demo_bq, demo_a)]

    df["Answer"] = df["Answer"].astype(str)
    df = _add_prompt_columns(df, SVAMP_COT_DEMO, nocot_demos, templates)

    df.to_json(output_path, orient="records", indent=4)
    print(f"SVAMP: {len(df)} examples -> {output_path}")
    return df


# ===========================================================================
# ProntoQA
# ===========================================================================

PRONTOQA_STRATIFY = ("hop",)


def curate_prontoqa_and_save_json(
    hop_files: Dict[int, str],
    output_path: str,
    templates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Reads the True/False QA files produced by prontoqa-main's run_experiment.py
    with --model-name json and WITHOUT --proofs-only.

    That flag matters: with --proofs-only the generator emits proof traces and no
    `answer` field at all, which is why the previously downloaded OOD dumps could
    not be used here.

    hop_files maps hop count -> path.
    """
    records, cot_demo = [], None

    for hop in sorted(hop_files):
        data = json.load(open(hop_files[hop], "r", encoding="utf-8"))
        for key, item in data.items():
            test = item["test_example"]
            demo = item.get("in_context_example0")

            question = _collapse(f"{test['question']} {test['query']}")
            demo_fields = None
            if demo is not None:
                demo_fields = {
                    "question": _collapse(f"{demo['question']} {demo['query']}"),
                    "reasoning": " ".join(demo["chain_of_thought"]),
                    "answer": str(demo["answer"]),
                }
                # the first in-context example becomes the shared CoT exemplar
                if cot_demo is None:
                    cot_demo = demo_fields

            records.append({
                "ID": f"{hop}hop-{key}",
                "hop": hop,
                "PromptWithoutExample": question,
                "Answer": str(test["answer"]),
                "ChainOfThought": " ".join(test["chain_of_thought"]),
                "_nocot_demo": demo_fields,
            })

    if cot_demo is None:
        raise ValueError("no in-context example found; regenerate with --few-shot-examples 1")

    df = pd.DataFrame(records)
    nocot_demos = [d if d is not None else cot_demo for d in df.pop("_nocot_demo")]
    df = _add_prompt_columns(df, cot_demo, nocot_demos, templates)

    df.to_json(output_path, orient="records", indent=4)
    print(f"ProntoQA: {len(df)} examples -> {output_path}")
    print(f"  hops: {df['hop'].value_counts().sort_index().to_dict()}")
    print(f"  answers: {df['Answer'].value_counts().to_dict()}")
    return df


# ===========================================================================
# Balanced subsets
# ===========================================================================

def create_and_save_balanced_subset(
    input_json_path: str,
    output_json_path: str,
    n_samples: int = 4,
    stratify_keys: Sequence[str] = SVAMP_STRATIFY,
    random_state: int = 42,
    drop_zero_operations: bool = True,
    keep_groups: Optional[Dict[str, Sequence]] = None,
) -> pd.DataFrame:
    """
    Samples n_samples rows from each stratification group.

    keep_groups restricts the pool before sampling, e.g. {"hop": (2, 3, 4, 5)}
    to exclude a difficulty level entirely rather than merely under-sample it.
    """
    df = pd.read_json(input_json_path)

    if drop_zero_operations and "OperationCount" in df.columns:
        df = df[df["OperationCount"] != 0]

    for col, allowed in (keep_groups or {}).items():
        if col not in df.columns:
            raise ValueError(f"keep_groups references missing column {col!r}")
        before = len(df)
        df = df[df[col].isin(list(allowed))]
        print(f"  keep_groups: {col} in {tuple(allowed)} -> {len(df)}/{before} rows")

    keys = [k for k in stratify_keys if k in df.columns]
    if not keys:
        raise ValueError(f"none of {tuple(stratify_keys)} present; have {list(df.columns)}")

    sampled = (
        df.groupby(list(keys), group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), n_samples), random_state=random_state))
          .reset_index(drop=True)
    )
    sampled.to_json(output_json_path, orient="records", indent=4)
    print(f"Balanced subset: {len(sampled)} examples "
          f"(<= {n_samples} per {'/'.join(keys)}) -> {output_json_path}")
    return sampled
