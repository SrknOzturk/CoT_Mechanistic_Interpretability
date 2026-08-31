"""
src/tasks.py

Task registry.

The patching pipeline was written around arithmetic: answers were parsed with a
number regex, compared with a float tolerance, and the token-by-token answer
loop only accepted digits. None of that transfers to a True/False task, so the
task-specific parts are collected here and everything else stays generic.

A TaskSpec answers four questions about a dataset:
    * where the identifier and gold answer live
    * how to turn generated text into an answer
    * how to decide whether two answers agree
    * which tokens still belong to the answer while it is being generated
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

ANSWER_TRIGGER = "The answer is "


class AnswerTriggerNotFound(RuntimeError):
    """
    Raised when a model never emits the answer trigger within its token budget.

    This is an expected outcome for a base model on a hard prompt, not a bug, so
    it gets its own type: the drivers catch it, record the example as skipped and
    carry on, while a genuine error still propagates.
    """

    def __init__(self, prompt_kind, budget, generated_tail=""):
        self.prompt_kind = prompt_kind
        self.budget = budget
        self.generated_tail = generated_tail
        super().__init__(
            f"{prompt_kind}: answer trigger not reached within {budget} tokens"
            + (f"; trace ended with {generated_tail!r}" if generated_tail else "")
        )


# ===========================================================================
# Answer parsing
# ===========================================================================

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOOL_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def extract_last_number(text: Any) -> Optional[float]:
    """Final numeric value in a generated block."""
    matches = _NUMBER_RE.findall(str(text))
    return float(matches[-1]) if matches else None


def extract_first_bool(text: Any) -> Optional[bool]:
    """
    First True/False in a generated block.

    First, not last: the answer is the token immediately after the trigger, and
    anything following it is the model running on into the next question.
    """
    m = _BOOL_RE.search(str(text))
    if not m:
        return None
    return m.group(1).lower() == "true"


def numeric_equal(pred: Any, gold: Any, tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < tol
    except (TypeError, ValueError):
        return False


def bool_equal(pred: Any, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    return _as_bool(pred) == _as_bool(gold)


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


# ===========================================================================
# Answer-continuation gates
#
# Used by the generation loop to decide whether the token just produced is still
# part of the answer, or the model has moved on to the next question.
# ===========================================================================

_NUMERIC_CHARS = set("0123456789.,-")


def numeric_continuation(token_str: Optional[str]) -> bool:
    """Accepts "38", " 38", ".", "0", "-", ","; rejects " Therefore", "\\n", " "."""
    if not token_str:
        return False
    stripped = token_str.strip()
    if not stripped:
        return False
    return all(ch in _NUMERIC_CHARS for ch in stripped)


def alpha_continuation(token_str: Optional[str]) -> bool:
    """
    Accepts the alphabetic pieces of True / False, however the tokenizer splits
    them ("True", " Tr", "ue"). Stops at punctuation, whitespace or a newline,
    which is where the model starts the next question.
    """
    if not token_str:
        return False
    stripped = token_str.strip()
    if not stripped:
        return False
    return stripped.isalpha()


# ===========================================================================
# TaskSpec
# ===========================================================================

@dataclass(frozen=True)
class TaskSpec:
    key: str
    parse_answer: Callable[[Any], Any]
    answers_equal: Callable[[Any, Any], bool]
    is_answer_continuation: Callable[[Optional[str]], bool]
    id_column: str = "ID"
    answer_column: str = "Answer"
    answer_trigger: str = ANSWER_TRIGGER
    answer_trigger_alts: Tuple[str, ...] = ()
    stratify_keys: Tuple[str, ...] = ()
    dataset_file: str = ""
    description: str = ""

    @property
    def triggers(self) -> Tuple[str, ...]:
        """Every string whose appearance ends the reasoning phase."""
        return (self.answer_trigger,) + tuple(self.answer_trigger_alts)

    def ends_reasoning(self, text: str) -> bool:
        return any(text.endswith(t) for t in self.triggers)

    def answer_segment(self, text: str) -> str:
        """
        The part of a generated string that follows the final answer trigger.

        Necessary because the question itself can contain answer-like tokens:
        every ProntoQA query reads "True or false: ...", so scanning the whole
        string finds the word in the prompt rather than the model's answer.

        Slicing uses the base trigger only, so a negative numeric answer keeps
        its sign when the model emitted "The answer is -".
        """
        idx = text.rfind(self.answer_trigger)
        return text[idx + len(self.answer_trigger):] if idx >= 0 else text

    def extract(self, text: str) -> Any:
        """Parses the answer out of a generated string."""
        return self.parse_answer(self.answer_segment(text))

    def gold_from_row(self, row) -> Any:
        for col in (self.answer_column, "Answer", "answer", "TargetAnswer", "target", "label"):
            if col in row and row[col] is not None:
                return row[col]
        raise KeyError(f"[{self.key}] no answer column found; looked for {self.answer_column!r}")

    def is_correct(self, generated_text: str, gold: Any) -> bool:
        return self.answers_equal(self.extract(generated_text), gold)


TASKS: Dict[str, TaskSpec] = {
    "svamp": TaskSpec(
        key="svamp",
        parse_answer=extract_last_number,
        answers_equal=numeric_equal,
        is_answer_continuation=numeric_continuation,
        # a leading minus arrives as part of the trigger rather than the answer
        answer_trigger_alts=("The answer is -",),
        stratify_keys=("OperationCount", "Type"),
        dataset_file="svamp_candidates.json",
        description="math word problems; numeric answers",
    ),
    "prontoqa": TaskSpec(
        key="prontoqa",
        parse_answer=extract_first_bool,
        answers_equal=bool_equal,
        is_answer_continuation=alpha_continuation,
        stratify_keys=("hop",),
        dataset_file="prontoqa_candidates.json",
        description="synthetic deduction over a fictional ontology; True/False answers",
    ),
}

DEFAULT_TASK = "svamp"


def get_task(key: str = DEFAULT_TASK) -> TaskSpec:
    if key not in TASKS:
        raise KeyError(f"unknown task {key!r}; available: {sorted(TASKS)}")
    return TASKS[key]
