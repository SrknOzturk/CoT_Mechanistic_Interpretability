"""
src/gpu_planning.py

How many copies of a model fit on the GPU, and what a run will cost.

Parallelism here is process-level, not batch-level: each worker loads its own
copy of the model and takes a different slice of the examples. Batching examples
together is not an option, because unequal prompt lengths force left-padding and
the patch site is defined as "the last position" -- padding would move it.
Running N independent single-example processes sidesteps that entirely.

It pays off because at batch size 1 a 0.5-1B model leaves the GPU badly
underused: a forward pass is dominated by kernel-launch overhead rather than
arithmetic, so concurrent processes fill each other's idle gaps.

Every number below is an estimate. `experiments/plan_compute.py` prints them so
the assumptions can be argued with, and the pre-filter replaces the step-count
guesses with measurements before any long run starts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.models import MODELS, ModelSpec, get_model_spec

BYTES_PER_GB = 1024 ** 3

# A CUDA context plus the PyTorch/TransformerLens runtime, per process.
PROCESS_OVERHEAD_GB = 0.9

# TransformerLens caches many activations per layer during run_with_cache.
# Counted as an equivalent number of [seq, d_model] tensors per layer.
CACHE_TENSORS_PER_LAYER = 12

# Measured on an RTX 6000 Ada: 29.58 s for one full 24x14 sweep on Qwen2.5-0.5B,
# i.e. 336 forward passes. Unsynchronised, so treat as approximate.
QWEN_SECONDS_PER_SWEEP = 29.58
QWEN_HEAD_SLOTS = 336
SECONDS_PER_PASS_QWEN = QWEN_SECONDS_PER_SWEEP / QWEN_HEAD_SLOTS

# At batch 1 these models are launch-bound, so per-pass time grows far more
# slowly than parameter count. This exponent maps the parameter ratio onto the
# per-pass time ratio; 1.0 would be fully compute-bound, 0.0 fully overhead-bound.
PARAM_SCALING_EXPONENT = 0.4

# Slicing the residual to the last position before ln_final/unembed removes an
# unembed matmul over the whole sequence. Exact, since both ops are positionwise.
LAST_POS_SLICING_SAVING = 0.22


@dataclass
class MemoryEstimate:
    weights_gb: float
    cache_gb: float
    logits_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.cache_gb + self.logits_gb + self.overhead_gb


def estimate_worker_memory(spec: ModelSpec, seq_len: int = 400,
                           dtype_bytes: int = 4) -> MemoryEstimate:
    """Peak GPU memory for one worker holding one model copy."""
    weights = spec.n_params * dtype_bytes / BYTES_PER_GB

    # activations kept by run_with_cache: residual-stream-shaped tensors,
    # attention patterns, and the per-head q/k/v/z tensors
    resid = CACHE_TENSORS_PER_LAYER * spec.n_layers * seq_len * spec.d_model
    patterns = spec.n_layers * spec.n_heads * seq_len * seq_len
    qkvz = 4 * spec.n_layers * seq_len * spec.d_model
    cache = (resid + patterns + qkvz) * dtype_bytes / BYTES_PER_GB

    # logits for a single forward pass; the dominant transient without slicing
    logits = seq_len * spec.d_vocab * dtype_bytes / BYTES_PER_GB

    return MemoryEstimate(weights, cache, logits, PROCESS_OVERHEAD_GB)


def max_workers(spec: ModelSpec, gpu_total_gb: float, gpu_used_gb: float = 0.0,
                seq_len: int = 400, safety_factor: float = 1.25,
                hard_cap: Optional[int] = None) -> int:
    """
    How many workers fit in the free VRAM, with headroom.

    safety_factor covers fragmentation and transient spikes; hard_cap lets the
    caller stop before the GPU saturates on compute rather than memory.
    """
    per_worker = estimate_worker_memory(spec, seq_len) .total_gb * safety_factor
    free = max(gpu_total_gb - gpu_used_gb, 0.0)
    n = int(free // per_worker)
    if hard_cap is not None:
        n = min(n, hard_cap)
    return max(n, 1)


def seconds_per_pass(spec: ModelSpec) -> float:
    """Per forward pass, relative to the measured Qwen baseline."""
    ratio = spec.n_params / get_model_spec("qwen2.5-0.5b").n_params
    return SECONDS_PER_PASS_QWEN * (ratio ** PARAM_SCALING_EXPONENT)


def seconds_per_token_step(spec: ModelSpec, slicing: bool = True) -> float:
    """One sequential step = one full sweep over every (layer, head)."""
    t = spec.n_head_slots * seconds_per_pass(spec)
    return t * (1 - LAST_POS_SLICING_SAVING) if slicing else t


@dataclass
class CellCost:
    model: str
    dataset: str
    n_examples: int
    mean_steps: int
    sequential_runs: int
    sweep_runs: int
    gpu_hours: float


def estimate_cell(spec: ModelSpec, dataset: str, n_examples: int, mean_steps: int,
                  sequential_runs: int = 1, sweep_runs: int = 2,
                  slicing: bool = True) -> CellCost:
    """
    sequential_runs: scans over the whole CoT trace (normal, cross)
    sweep_runs     : single-position controls (the random baselines)
    """
    step = seconds_per_token_step(spec, slicing)
    seconds = n_examples * (sequential_runs * mean_steps + sweep_runs) * step
    return CellCost(spec.key, dataset, n_examples, mean_steps,
                    sequential_runs, sweep_runs, seconds / 3600)


def wall_clock_hours(gpu_hours: float, workers: int, efficiency: float = 0.65) -> float:
    """
    Concurrent workers do not scale linearly: they contend for SMs, memory
    bandwidth and the scheduler. `efficiency` is the fraction of ideal speedup
    actually realised, and should be replaced by a measurement.
    """
    return gpu_hours / max(workers * efficiency, 1.0)
