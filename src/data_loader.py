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

SVAMP preserves the original design: the CoT demonstration is one fixed
exemplar, while the No-CoT demonstration is drawn per row and matched on the
operation type.  ProntoQA uses one fixed exemplar on both sides so that the
clean/corrupted pair differs in reasoning content rather than demonstration
identity.
"""

import json
import re
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


def _select_reasoning(demo, key: str):
    """
    Returns the demonstration(s) with "reasoning" set to the requested writing.

    Missing is an error rather than a fallback: silently showing a step-by-step
    demonstration under a plan-and-solve cue would run the experiment with the
    wrong prompt and nothing downstream would notice.
    """
    if key == "reasoning":
        return demo
    items = demo if isinstance(demo, list) else [demo]
    out = []
    for item in items:
        if key not in item:
            raise KeyError(
                f"demonstration has no {key!r}; add it next to 'reasoning' for this dataset"
            )
        out.append({**item, "reasoning": item[key]})
    return out if isinstance(demo, list) else out[0]


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
        # a plan-and-solve cue needs a demonstration that shows a plan; the
        # template names which writing of the same demonstration it wants
        demo_for_spec = _select_reasoning(cot_demo, spec.reasoning_key)
        for (_, row), nocot_demo in zip(df.iterrows(), nocot_demos):
            target = {"question": row["PromptWithoutExample"]}
            cot_col.append(spec.render_cot(target, demo_for_spec))
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
    # The same demonstration written for the PS+ cue: the Plan: / Solution:
    # layout of Figure 2 in Wang et al. 2023, plus the variable extraction the
    # cue asks for.
    "reasoning_plan_solve_plus": (
        "Variables: balls Roger started with = 5, cans bought = 2, balls per can = 3. "
        "Plan: Step 1: Work out how many tennis balls are in the cans. "
        "Step 2: Add those balls to the ones Roger already had. "
        "Solution: Step 1: 2 cans * 3 balls per can = 6 balls. "
        "Step 2: 5 balls + 6 balls = 11 balls."
    ),
    "answer": "11",
}

SVAMP_STRATIFY = ("OperationCount", "Type")

# Plan-and-Solve is being trialled on SVAMP only; rendering it elsewhere would
# add prompt columns no run reads and demand a plan-shaped demonstration the
# other datasets have no reason to define.
SVAMP_TEMPLATES = ("step_by_step", "qa1shot", "plan_solve_plus")
PRONTOQA_TEMPLATES = ("step_by_step", "qa1shot")


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
    df = _add_prompt_columns(df, SVAMP_COT_DEMO, nocot_demos, templates or SVAMP_TEMPLATES)

    df.to_json(output_path, orient="records", indent=4)
    print(f"SVAMP: {len(df)} examples -> {output_path}")
    return df


# ===========================================================================
# ProntoQA
# ===========================================================================

PRONTOQA_STRATIFY = ("hop",)
PRONTOQA_BINARY_INSTRUCTION = "True or false:"


def _prontoqa_binary_label(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return "True"
    if normalized in {"false", "0"}:
        return "False"
    raise ValueError(f"unexpected ProntoQA answer label: {value!r}")


def _prontoqa_binary_question(question: str, query: str) -> str:
    """Render a ProntoQA query with an explicit True/False question."""
    statement = re.sub(
        r"^\s*true\s+or\s+false\s*:\s*",
        "",
        str(query),
        flags=re.IGNORECASE,
    )
    return _collapse(f"{question} {PRONTOQA_BINARY_INSTRUCTION} {statement}")


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
    records, demos_by_label = [], {}

    for hop in sorted(hop_files):
        data = json.load(open(hop_files[hop], "r", encoding="utf-8"))
        for key, item in data.items():
            test = item["test_example"]
            demo = item.get("in_context_example0")

            question = _prontoqa_binary_question(test["question"], test["query"])
            if demo is not None:
                demo_fields = {
                    "question": _prontoqa_binary_question(demo["question"], demo["query"]),
                    "reasoning": " ".join(demo["chain_of_thought"]),
                    "answer": _prontoqa_binary_label(demo["answer"]),
                }
                demos_by_label.setdefault(demo_fields["answer"], demo_fields)

            records.append({
                "ID": f"{hop}hop-{key}",
                "hop": hop,
                "PromptWithoutExample": question,
                "Answer": _prontoqa_binary_label(test["answer"]),
                "ChainOfThought": " ".join(test["chain_of_thought"]),
            })

    if set(demos_by_label) != {"False", "True"}:
        raise ValueError("need at least one False-demo and one True-demo")

    # A fixed balanced pair prevents the model from learning that the only
    # demonstrated answer is False. Keep the same examples and order on both sides.
    cot_demo = [demos_by_label["False"], demos_by_label["True"]]

    df = pd.DataFrame(records)
    # Keep the 1-shot demonstration identical across the clean CoT and corrupt
    # No-CoT prompts.  Only the reasoning cue/trace is removed on the No-CoT
    # side; changing the exemplar as well would confound activation patching.
    nocot_demos = [cot_demo] * len(df)
    df = _add_prompt_columns(df, cot_demo, nocot_demos, templates or PRONTOQA_TEMPLATES)

    df.to_json(output_path, orient="records", indent=4)
    print(f"ProntoQA: {len(df)} examples -> {output_path}")
    print(f"  hops: {df['hop'].value_counts().sort_index().to_dict()}")
    print(f"  answers: {df['Answer'].value_counts().to_dict()}")
    return df


# ===========================================================================
# BIG-bench Boolean Expressions
# ===========================================================================

# These are the length-4/5/6 demonstrations selected for the experiment.  Keep
# the prose and line breaks in one shared definition so the pilot, patching and
# ablation paths cannot silently drift onto different prompts.
BIGBENCH_BOOLEAN_COT_PROMPT = """Evaluate the Boolean expression. Evaluate parentheses first, then not, then and, then or.

Q: not True and True is
A: Let's think step by step. not True = False. Therefore False and True = False. The answer is False.

Q: True and False or True is
A: Let's think step by step. True and False = False. Therefore False or True = True. The answer is True.

Q: not ( False ) and False is
A: Let's think step by step. not False = True. Therefore True and False = False. The answer is False."""

BIGBENCH_BOOLEAN_NOCOT_PROMPT = """Evaluate the Boolean expression. Answer True or False.

Q: not True and True is
A: The answer is False.

Q: True and False or True is
A: The answer is True.

Q: not ( False ) and False is
A: The answer is False."""

BIGBENCH_BOOLEAN_DEMO_QUESTIONS = (
    "not True and True is",
    "True and False or True is",
    "not ( False ) and False is",
)


def _boolean_target_question(value: str) -> str:
    """Normalize a BIG-bench input such as ``'True and False is '``."""
    return _collapse(value).strip()


def curate_bigbench_boolean_expressions_and_save_json(
    source_json_path: str,
    output_path: str,
    reserve_json_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert the fixed balanced Boolean Expressions set to the flat prompt table
    consumed by the existing patching and ablation runners.

    The default ``step_by_step`` columns reproduce the exact prompt used in the
    Boolean pilot.  ``qa1shot`` columns are also populated for CLI compatibility,
    but new runs should use the default template.
    """
    def load_rows(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("records", payload)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Boolean Expressions source has no records: {path}")
        return rows

    def convert_rows(rows: List[Dict], pool_split: str) -> List[Dict]:
        records = []
        for row in rows:
            answer = str(row["target"]).strip().title()
            if answer not in {"True", "False"}:
                raise ValueError(f"unexpected Boolean Expressions answer: {row['target']!r}")
            question = _boolean_target_question(row["input"])
            records.append({
                "ID": str(row["id"]),
                "source": row.get("source", "BIG-bench boolean_expressions"),
                "pool_split": pool_split,
                "length": int(row["length"]),
                "Type": f"length_{int(row['length'])}",
                "PromptWithoutExample": question,
                "Answer": answer,
                "seed": row.get("seed"),
                "possible_expressions_at_length": row.get("possible_expressions_at_length"),
            })
        return records

    primary_records = convert_rows(load_rows(source_json_path), "primary")
    reserve_records = (
        convert_rows(load_rows(reserve_json_path), "reserve")
        if reserve_json_path else []
    )
    df = pd.DataFrame(primary_records + reserve_records)

    if df["ID"].duplicated().any():
        raise ValueError("duplicate IDs across the Boolean primary and reserve pools")
    if df["PromptWithoutExample"].duplicated().any():
        raise ValueError("duplicate questions across the Boolean primary and reserve pools")
    demo_questions = {_boolean_target_question(q) for q in BIGBENCH_BOOLEAN_DEMO_QUESTIONS}
    overlap = set(df["PromptWithoutExample"]) & demo_questions
    if overlap:
        raise ValueError(f"demonstration questions leaked into the evaluation pool: {overlap}")

    step = get_template("step_by_step")
    df[step.cot_col] = [
        f"{BIGBENCH_BOOLEAN_COT_PROMPT}\n\nQ: {q}\nA: Let's think step by step."
        for q in df["PromptWithoutExample"]
    ]
    df[step.nocot_col] = [
        f"{BIGBENCH_BOOLEAN_NOCOT_PROMPT}\n\nQ: {q}\nA:"
        for q in df["PromptWithoutExample"]
    ]

    # Retain the legacy template as an explicitly selectable reproduction
    # option without changing its behavior for existing datasets.
    qa1 = get_template("qa1shot")
    demos = [
        {
            "question": BIGBENCH_BOOLEAN_DEMO_QUESTIONS[0],
            "reasoning": "not True = False. Therefore False and True = False.",
            "answer": "False",
        },
        {
            "question": BIGBENCH_BOOLEAN_DEMO_QUESTIONS[1],
            "reasoning": "True and False = False. Therefore False or True = True.",
            "answer": "True",
        },
        {
            "question": BIGBENCH_BOOLEAN_DEMO_QUESTIONS[2],
            "reasoning": "not False = True. Therefore True and False = False.",
            "answer": "False",
        },
    ]
    df[qa1.cot_col] = [
        "Evaluate the Boolean expression. Evaluate parentheses first, then not, then and, then or.\n\n"
        + qa1.render_cot({"question": q}, demos)
        for q in df["PromptWithoutExample"]
    ]
    df[qa1.nocot_col] = [
        "Evaluate the Boolean expression. Answer True or False.\n\n"
        + qa1.render_nocot({"question": q}, demos)
        for q in df["PromptWithoutExample"]
    ]

    primary_df = df[df["pool_split"] == "primary"]
    counts = primary_df.groupby(["length", "Answer"]).size().to_dict()
    expected = {(length, label): 10 for length in (4, 5, 6) for label in ("False", "True")}
    if counts != expected:
        raise ValueError(f"expected 10 examples per length/answer cell, found {counts}")

    if reserve_records:
        reserve_df = df[df["pool_split"] == "reserve"]
        reserve_counts = reserve_df.groupby(["length", "Answer"]).size().to_dict()
        expected_reserve = {
            (5, "False"): 7,
            (5, "True"): 8,
            (6, "False"): 8,
            (6, "True"): 7,
        }
        if reserve_counts != expected_reserve:
            raise ValueError(
                f"unexpected Boolean reserve length/answer counts: {reserve_counts}"
            )

    df.to_json(output_path, orient="records", indent=4, force_ascii=False)
    print(
        "BIG-bench Boolean Expressions: "
        f"{len(primary_records)} primary + {len(reserve_records)} reserve -> {output_path}"
    )
    return df


# ===========================================================================
# BIG-bench Web of Lies
# ===========================================================================

# The official BBH examples have five statements, which makes OLMo continue
# inventing extra statements on the two-statement evaluation items.  These two
# demonstrations match the selected task length, cover both outcomes, and are
# shared by every model and every example.
WEB_OF_LIES_COT_PROMPT = """Evaluate a random boolean function expressed as a word problem.

Q: Question: Fidel tells the truth. Jerry says Fidel tells the truth. Does Jerry tell the truth?
A: Let's think step by step.
(1) Fidel tells the truth, so Fidel tells the truth.
(2) Jerry says Fidel tells the truth. This is correct, so Jerry tells the truth.
The answer is Yes.

Q: Question: Kristian lies. Leda says Kristian tells the truth. Does Leda tell the truth?
A: Let's think step by step.
(1) Kristian lies, so Kristian does not tell the truth.
(2) Leda says Kristian tells the truth. This is incorrect, so Leda lies.
The answer is No."""

WEB_OF_LIES_NOCOT_PROMPT = """Evaluate a random boolean function expressed as a word problem. Answer Yes or No.

Q: Question: Fidel tells the truth. Jerry says Fidel tells the truth. Does Jerry tell the truth?
A: The answer is Yes.

Q: Question: Kristian lies. Leda says Kristian tells the truth. Does Leda tell the truth?
A: The answer is No."""


def curate_bigbench_web_of_lies_and_save_json(
    source_json_path: str,
    output_path: str,
    reserve_json_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create the flat candidate table consumed by patching and ablation.

    ``source_json_path`` must be the fixed 64-question set produced by
    ``prepare_bigbench_web_of_lies_datasets.py``.  The optional 32-question
    reserve is appended after it so ``run_parallel.py`` can replace a target
    item only when that model fails to reach the answer anchor.
    """
    def load_rows(path: str) -> List[Dict]:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("examples")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Web of Lies source has no examples: {path}")
        return rows

    def convert_rows(rows: List[Dict], pool_split: str) -> List[Dict]:
        records = []
        for row in rows:
            answer = str(row["gold_answer"]).strip().title()
            if answer not in {"Yes", "No"}:
                raise ValueError(f"unexpected Web of Lies answer: {row['gold_answer']!r}")
            if int(row.get("chain_length", 0)) != 2:
                raise ValueError("Web of Lies patching candidates must have two statements")
            records.append({
                "ID": str(row["id"]),
                "source": "BIG-bench Web of Lies official two-statement generator",
                "pool_split": pool_split,
                "truth_pattern": str(row["truth_pattern"]),
                "first_statement_truth": bool(row["first_statement_truth"]),
                "second_statement_truth": bool(row["second_statement_truth"]),
                "chain_length": 2,
                "Type": str(row["truth_pattern"]),
                "PromptWithoutExample": _collapse(row["input"]),
                "Answer": answer,
            })
        return records

    primary_records = convert_rows(load_rows(source_json_path), "primary")
    reserve_records = (
        convert_rows(load_rows(reserve_json_path), "reserve")
        if reserve_json_path else []
    )
    df = pd.DataFrame(primary_records + reserve_records)

    if df["ID"].duplicated().any():
        raise ValueError("duplicate Web of Lies IDs across primary and reserve pools")
    if df["PromptWithoutExample"].duplicated().any():
        raise ValueError("duplicate Web of Lies questions across primary and reserve pools")

    primary_counts = df[df["pool_split"] == "primary"].groupby("truth_pattern").size().to_dict()
    expected_primary = {
        "first_true__second_true": 16,
        "first_true__second_false": 16,
        "first_false__second_true": 16,
        "first_false__second_false": 16,
    }
    if primary_counts != expected_primary:
        raise ValueError(f"expected 16 primary rows per truth pattern, found {primary_counts}")
    primary_answers = df[df["pool_split"] == "primary"]["Answer"].value_counts().to_dict()
    if primary_answers != {"Yes": 32, "No": 32}:
        raise ValueError(f"expected a 32/32 primary answer split, found {primary_answers}")

    if reserve_records:
        reserve_counts = df[df["pool_split"] == "reserve"].groupby("truth_pattern").size().to_dict()
        expected_reserve = {key: 8 for key in expected_primary}
        if reserve_counts != expected_reserve:
            raise ValueError(f"expected 8 reserve rows per truth pattern, found {reserve_counts}")

    # The generic runners select a TemplateSpec by CLI name.  These prompts
    # are deliberately written rather than rendered by the generic templates:
    # both demonstrations must remain length-matched to the two-statement task
    # and end with the pipeline's standard "The answer is" anchor.
    for template_key in ("step_by_step", "qa1shot"):
        template = get_template(template_key)
        df[template.cot_col] = [
            f"{WEB_OF_LIES_COT_PROMPT}\n\nQ: Question: {question}\nA: Let's think step by step."
            for question in df["PromptWithoutExample"]
        ]
        df[template.nocot_col] = [
            f"{WEB_OF_LIES_NOCOT_PROMPT}\n\nQ: Question: {question}\nA: The answer is "
            for question in df["PromptWithoutExample"]
        ]

    df.to_json(output_path, orient="records", indent=4, force_ascii=False)
    print(
        "BIG-bench Web of Lies: "
        f"{len(primary_records)} primary + {len(reserve_records)} reserve -> {output_path}"
    )
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

    # pandas 3 no longer includes grouping columns in DataFrameGroupBy.apply's
    # group frames.  Concatenating samples from the groups directly keeps the
    # stratification columns in the persisted candidate pool on every supported
    # pandas version.
    sampled = pd.concat(
        [
            group.sample(n=min(len(group), n_samples), random_state=random_state)
            for _, group in df.groupby(list(keys), sort=True, dropna=False)
        ],
        ignore_index=True,
    )
    sampled.to_json(output_json_path, orient="records", indent=4)
    print(f"Balanced subset: {len(sampled)} examples "
          f"(<= {n_samples} per {'/'.join(keys)}) -> {output_json_path}")
    return sampled
