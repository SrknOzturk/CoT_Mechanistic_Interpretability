"""
src/models.py

Model registry.

All models run in float32 so precision is never a confound when comparing them.
Each entry records the head geometry, because the cost of a sequential patching
run scales with n_layers * n_heads (one forward pass per head, per token step).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    tl_name: str
    dtype: torch.dtype = torch.float32
    gated: bool = False
    # documentation only -- the authoritative values come from model.cfg
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    d_model: Optional[int] = None
    d_vocab: Optional[int] = None
    n_params: Optional[float] = None   # total parameters, for the memory estimate
    note: str = ""

    @property
    def n_head_slots(self) -> Optional[int]:
        """Forward passes per token step in a full sweep."""
        if self.n_layers is None or self.n_heads is None:
            return None
        return self.n_layers * self.n_heads


MODELS: Dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        tl_name="qwen2.5-0.5b",
        n_layers=24, n_heads=14, d_model=896, d_vocab=151936, n_params=0.494e9,
        note="base model of the submitted version; produces CoT only when prompted",
    ),
    "olmo2-1b": ModelSpec(
        key="olmo2-1b",
        tl_name="allenai/OLMo-2-0425-1B",
        n_layers=16, n_heads=16, d_model=2048, d_vocab=100352, n_params=1.48e9,
        note="added during the EMNLP rebuttal",
    ),
    "llama3.2-1b": ModelSpec(
        key="llama3.2-1b",
        tl_name="meta-llama/Llama-3.2-1B",
        gated=True,
        n_layers=16, n_heads=32, d_model=2048, d_vocab=128256, n_params=1.24e9,
        note="base, not Instruct: an instruction-tuned model reasons even under "
             "the No-CoT prompt, which would collapse the CoT/No-CoT contrast",
    ),
}

DEFAULT_MODEL = "qwen2.5-0.5b"


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {sorted(MODELS)}")
    return MODELS[key]


def relative_cost(key: str, reference: str = DEFAULT_MODEL) -> Optional[float]:
    """
    Rough cost of one cell relative to the reference model: forward passes per
    token step, which is the term that actually differs between these models.
    Wall-clock also scales with parameter count, so treat this as a lower bound.
    """
    a, b = get_model_spec(key).n_head_slots, get_model_spec(reference).n_head_slots
    if a is None or b is None:
        return None
    return a / b
