# Mechanistic Interpretability of Chain-of-Thought Reasoning in Math Word Problems

This repository contains the official implementation for the paper: **"Mechanistic Interpretability of Chain-of-Thought Reasoning in Math Word Problems: CoT as a Meta-Controller"**.

## 🚀 Overview
In this work, we hypothesize that Chain-of-Thought (CoT) prompting acts as a **meta-controller** within the network, dynamically coordinating and activating specific problem-solving sub-circuits. We introduce **Sequential Multi-Head Patching**, a novel interpretability method that maps these functional circuits by analyzing activation flow across multiple tokens.

Models, datasets and prompt templates are registry entries rather than code paths, so adding one does not mean adding a new pipeline:

| Registry | Entries |
|---|---|
| Models (`src/models.py`) | `qwen2.5-0.5b`, `olmo2-1b`, `llama3.2-1b` (all base, all float32) |
| Datasets (`src/tasks.py`) | `svamp` (numeric), `prontoqa` (True/False), `bigbench_boolean_expressions` (balanced lengths 4/5/6) |
| Templates (`src/templates.py`) | `step_by_step` (default), `qa1shot` |

## 📂 Repository Structure
```text
.
├── data/                       # raw, curated and candidate datasets
├── experiments/                # runnable entry points (see Workflow below)
│   ├── run_parallel.py         # main driver: patching + ablation, process-parallel
│   ├── run_ablation_parallel.py# ablation only, for an already-completed patching run
│   ├── run_k_pipeline.py       # resume-safe normal -> k-sweep -> chosen-k workflow
│   ├── rescore_k_sweep.py      # re-score saved heatmaps at other k values
│   ├── run_patchings.py        # the experiment functions themselves (also a CLI)
│   └── run_ablations.py        # sequential ablation (superseded by the parallel path)
├── results/                    # JSON/CSV outputs from experiment runs
├── src/                        # Core implementation logic:
│   ├── ablation.py             # zero-ablation hooks, No-CoT/CoT/Direct-Equation runners
│   ├── analysis.py             # offline stats: bootstrap CIs, paired tests (CPU only)
│   ├── data_loader.py          # dataset handling and pre-processing
│   ├── gpu_planning.py         # worker-count sizing from available VRAM
│   ├── metrics.py              # JSD and Margin Recovery Ratio factories
│   ├── models.py               # model registry
│   ├── patching.py             # activation patching primitives
│   ├── patching_pipelines.py   # orchestration of a single layer x head sweep
│   ├── tasks.py                # per-dataset answer parsing/comparison
│   ├── templates.py            # prompt layout; owns the CoT/No-CoT pair
│   ├── utils.py                # generation, decoding penalties, NER/POS helpers
│   └── visualization.py        # heatmap plotting
├── prepare_dataset.py          # builds data/processed/*_candidates.json
├── requirements.txt            # Project dependencies
└── README.md

```

## 🛠 Setup & Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <project-directory>

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure NLP tools:**
```bash
python -m spacy download en_core_web_sm

```

4. **Build the candidate sets** (writes `data/processed/*_candidates.json`, which every
   experiment reads):
```bash
python prepare_dataset.py

```

`llama3.2-1b` is a gated repository; run `hf auth login` before using it.


## 📊 Workflow

### Running one model on one dataset

`run_parallel.py` is the main entry point. A single invocation runs both patching
experiments and, unless `--no-ablation` is passed, the ablation that verifies them:

```bash
python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp --target-n 64
```

The fixed Boolean Expressions pool keeps 60 primary questions and 30 automatic
reserves from lengths 5/6. The reserves replace primary examples that do not
reach the answer anchor. The selected length-4/5/6 demonstrations are already
included in the prompts, so the dataset parameter is the only task-specific
argument:

```bash
python experiments/run_parallel.py --dataset bigbench_boolean_expressions
```

What that covers, for one (model, dataset, template):

| Stage | Runs |
|---|---|
| Patching | `normal` (sequential multi-head patching) and `random` (Gaussian-noise control), each scoring **margin and JSD from one scan** → 4 result files |
| Ablation | every patching output × every metric × conditions No-CoT and CoT → 8 runs |

Each ablation run scores three ways per example: unablated, selected heads zeroed,
and an equally sized random head set zeroed (the control).

Work is sharded across worker processes, checkpointed per example, and retried with
fewer workers on CUDA OOM — re-running the same command resumes rather than
recomputes. Omit `--workers` to size the pool from available VRAM;
`python experiments/plan_compute.py` estimates cost beforehand.

**Direct-Equation ablation (SVAMP only).** `--equation-ablation` adds a third
condition whose prompt is the arithmetic alone (`76.0 - 25.0 = The answer is `),
with no question text and no reasoning cue, controlled against a layer-matched
random head set. It reads the `Equation` column, so it is rejected on ProntoQA:

```bash
python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp --target-n 64 --equation-ablation
```

**Decoding penalties (needed for ProntoQA).** Base models fall into verbatim loops
on ProntoQA's long deductive prompts, and a looping trace never reaches the answer
trigger — the example is then discarded as a skip rather than measured.
`--repetition-penalty` and `--no-repeat-ngram-size` are off by default, so SVAMP
results stay bit-identical, and apply to the reasoning phase only:

```bash
python experiments/run_parallel.py --model qwen2.5-0.5b --dataset prontoqa --target-n 64 \
    --repetition-penalty 1.15 --no-repeat-ngram-size 8
```

Values are a starting point, not a tuned setting — check the skip count in the run's
output and adjust. Whatever is used here must be repeated for `rescore_k_sweep.py`,
which regenerates the clean trace and rejects it if it differs from the saved one.

Other knobs: `--ctx`, `--heads-per-pos`, `--max-steps` (patching sweep depth),
`--ablation-max-new-tokens` (generation budget), `--seed`, `--fresh`, `--dry-run`.
To re-run only ablation against a completed patching run, use
`run_ablation_parallel.py`, which takes the same flags.

### Running the full study

Three models × two datasets. Each command is independently resumable:

```bash
# SVAMP
python experiments/run_parallel.py --model qwen2.5-0.5b --dataset svamp --target-n 64 --equation-ablation
python experiments/run_parallel.py --model olmo2-1b     --dataset svamp --target-n 64 --equation-ablation
python experiments/run_parallel.py --model llama3.2-1b  --dataset svamp --target-n 64 --equation-ablation

# ProntoQA
python experiments/run_parallel.py --model qwen2.5-0.5b --dataset prontoqa --target-n 64 --repetition-penalty 1.15 --no-repeat-ngram-size 8
python experiments/run_parallel.py --model olmo2-1b     --dataset prontoqa --target-n 64 --repetition-penalty 1.15 --no-repeat-ngram-size 8
python experiments/run_parallel.py --model llama3.2-1b  --dataset prontoqa --target-n 64 --repetition-penalty 1.15 --no-repeat-ngram-size 8
```

Models run one at a time: a worker pool is sized per model, and what runs
concurrently is examples of the *same* model.

### Resume-safe k workflow

`run_parallel.py` selects heads at one fixed budget (`--heads-per-pos`). This
workflow instead reuses the saved heatmaps to re-derive the selection at many `k`
values without repeating the expensive sweep. It first reuses or completes normal
patching, then scores every `k` from 1 through 30:

```bash
python experiments/run_k_pipeline.py prepare --model olmo2-1b --dataset svamp --workers 1 --gpu-total-gb 48
```

After inspecting the k-sweep outputs under `results/k_sweep`, choose the k for
each metric. This command runs one shared random-activation scan and then runs
CoT and No-CoT ablation for both the normal and random selected heads:

```bash
python experiments/run_k_pipeline.py complete --model olmo2-1b --dataset svamp --margin-k 6 --jsd-k 15 --workers 1 --gpu-total-gb 48
```

Every stage uses checkpoints and existing result files. Re-running either
command resumes missing examples and keeps earlier default-k results intact.

> `run_k_pipeline.py` does not yet forward `--repetition-penalty` /
> `--no-repeat-ngram-size` to the stages it calls. On ProntoQA, invoke
> `rescore_k_sweep.py` directly with the same penalties the patching run used, or
> its clean-trace check will reject the regenerated trace.

### Sequential entry points

`run_patchings.py` and `run_ablations.py` expose the same experiments without
process parallelism. They are kept for reference and for debugging a single
example; the parallel path above produces the same output files.

## 🔍 Methodology

* **Sequential Multi-Head Patching:** Unlike naive single-token intervention, our approach tracks the causal influence of heads across the full reasoning trace.
* **Meta-Controller Hypothesis:** We investigate how CoT tokens coordinate activation flow to calculation and abstraction sub-circuits.

