"""
src/templates.py

Prompt structure registry.

A template decides only how the CoT and No-CoT prompts are laid out; it never
changes which examples are used or what the demonstration says. Both sides keep
the same answer anchor ("The answer is ") so the patching site stays at the same
semantic position across templates and datasets.

The CoT and No-CoT prompts form the clean/corrupted pair of every patching run,
so the one asymmetry between them must be the chain-of-thought cue itself.
"""

from dataclasses import dataclass
from typing import Callable, Dict

from src.tasks import ANSWER_TRIGGER

STEP_BY_STEP = "Let's think step by step."


@dataclass(frozen=True)
class TemplateSpec:
    """
    render_cot / render_nocot take (target, demo) dicts, each with keys:
        question : str  -- the full base question (context + query)
        reasoning: str  -- the demonstration's chain of thought (demo only)
        answer   : str  -- the demonstration's answer (demo only)

    corrupt_suffix is appended to the No-CoT prompt to form the corrupted input;
    the result must end with ANSWER_TRIGGER.
    """
    key: str
    description: str
    corrupt_suffix: str
    render_cot: Callable[[Dict, Dict], str]
    render_nocot: Callable[[Dict, Dict], str]

    @property
    def cot_col(self) -> str:
        return f"PromptWithCot__{self.key}"

    @property
    def nocot_col(self) -> str:
        return f"PromptWithoutCot__{self.key}"


# ---------------------------------------------------------------------------
# step_by_step -- the structure used for the current round of experiments.
#
#   CoT   : Q: <demo> A: Let's think step by step. <reasoning> The answer is <a>.
#           Q: <target> A: Let's think step by step.
#   No-CoT: Q: <demo> A: The answer is <a>.
#           Q: <target> A:
#
# The cue appears only on the CoT side. That asymmetry is the manipulation.
# ---------------------------------------------------------------------------

def _cot_step_by_step(target: Dict, demo: Dict) -> str:
    return (
        f"Q: {demo['question']} "
        f"A: {STEP_BY_STEP} {demo['reasoning']} {ANSWER_TRIGGER}{demo['answer']}. "
        f"Q: {target['question']} "
        f"A: {STEP_BY_STEP}"
    )


def _nocot_step_by_step(target: Dict, demo: Dict) -> str:
    return (
        f"Q: {demo['question']} "
        f"A: {ANSWER_TRIGGER}{demo['answer']}. "
        f"Q: {target['question']} "
        f"A:"
    )


# ---------------------------------------------------------------------------
# qa1shot -- the structure used for the submitted version. Retained so the
# refactor can be checked against the existing results, not for new runs.
# ---------------------------------------------------------------------------

def _cot_qa1shot(target: Dict, demo: Dict) -> str:
    return (
        f"Q: {demo['question']} "
        f"A: {demo['reasoning']} {ANSWER_TRIGGER}{demo['answer']}. "
        f"Q: {target['question']} "
        f"A:"
    )


def _nocot_qa1shot(target: Dict, demo: Dict) -> str:
    return _nocot_step_by_step(target, demo)


TEMPLATES: Dict[str, TemplateSpec] = {
    "step_by_step": TemplateSpec(
        key="step_by_step",
        description="1-shot with an explicit 'Let's think step by step.' cue on the CoT side",
        corrupt_suffix=" " + ANSWER_TRIGGER,
        render_cot=_cot_step_by_step,
        render_nocot=_nocot_step_by_step,
    ),
    "qa1shot": TemplateSpec(
        key="qa1shot",
        description="original submitted structure, kept for reproduction checks only",
        corrupt_suffix=" " + ANSWER_TRIGGER,
        render_cot=_cot_qa1shot,
        render_nocot=_nocot_qa1shot,
    ),
}

DEFAULT_TEMPLATE = "step_by_step"


def get_template(key: str = DEFAULT_TEMPLATE) -> TemplateSpec:
    if key not in TEMPLATES:
        raise KeyError(f"unknown template {key!r}; available: {sorted(TEMPLATES)}")
    return TEMPLATES[key]


def check_template(spec: TemplateSpec, target: Dict, demo: Dict) -> None:
    """
    Guards the two invariants every patching run depends on. Cheap, and catches
    a malformed template before it costs GPU hours.
    """
    cot = spec.render_cot(target, demo)
    nocot = spec.render_nocot(target, demo)

    n = cot.count(ANSWER_TRIGGER)
    if n != 1:
        raise ValueError(
            f"[{spec.key}] CoT prompt must contain {ANSWER_TRIGGER!r} exactly once "
            f"before generation (found {n})"
        )
    corrupted = nocot + spec.corrupt_suffix
    if not corrupted.endswith(ANSWER_TRIGGER):
        raise ValueError(
            f"[{spec.key}] No-CoT prompt + corrupt_suffix must end with {ANSWER_TRIGGER!r}; "
            f"got {corrupted[-40:]!r}"
        )
    if STEP_BY_STEP in nocot:
        raise ValueError(
            f"[{spec.key}] the chain-of-thought cue leaked into the No-CoT prompt, "
            f"which would erase the difference between the two conditions"
        )
