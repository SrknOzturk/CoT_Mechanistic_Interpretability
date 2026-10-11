# Mechanistic Interpretability of CoT via Sequential Activation Patching

This repository contains the implementation and experiment scripts for the paper **“Mechanistic Interpretability of CoT via Sequential Activation Patching.”** The project studies how activations produced during Chain-of-Thought (CoT) generation influence a model's final-answer computation.

## Overview

The method has three main stages:

1. **Sequential single-head localization.** For each generated CoT token position, patch each attention head's clean-run activation (\`hook_z\`) individually into the matched No-CoT execution at the final-answer position. Evaluate each intervention against the clean CoT reference.
2. **POS-guided selection and joint patching.** Group token-level results by part-of-speech (POS) category. For each metric, select high-scoring heads within each category, deduplicate repeated layer–head pairs while retaining the strongest individual score and its associated source activation, and jointly patch the resulting set.
3. **Zero ablation.** Zero the selected heads throughout generation under CoT and No-CoT prompts. Compare the resulting accuracy with the unablated baseline and an equally sized random-head control.

A **Random Activation Patching** control replaces CoT-derived source activations with Gaussian random vectors while matching the number of jointly patched heads to the corresponding Sequential Activation Patching run.

The two main metrics are **Jensen–Shannon divergence (JSD)** and **Margin Recovery Ratio (MRR)**. Lower JSD and higher MRR indicate stronger alignment with the clean CoT reference. In result filenames, the implementation uses \`margin\` for MRR.

## Models and tasks

The main paper evaluates three base models and three reasoning tasks.

| Model key | Model | Notes |
|---|---|---|
| \`qwen2.5-0.5b\` | Qwen2.5-0.5B | Base model |
| \`olmo2-1b\` | OLMo-2-0425-1B | Base model |
| \`llama3.2-1b\` | Llama-3.2-1B | Base model |

| Dataset key | Task |
|---|---|
| \`svamp\` | Arithmetic word problems |
| \`bigbench_boolean_expressions\` | BIG-bench Boolean Expressions |
| \`bigbench_web_of_lies\` | BIG-bench Web of Lies |

The code also retains support for \`prontoqa\`, an additional synthetic deduction task, but it is not one of the three tasks in the main paper's evaluation.

The default prompt template is \`step_by_step\`, which uses a one-shot demonstration and an explicit CoT cue on the clean side. The alternative \`plan_solve_plus\` template is intended for SVAMP experiments; \`qa1shot\` is retained for reproduction checks.

## Repository structure

\`\`\`text
.
├── data/
│   ├── raw/                    # Source datasets
│   └── processed/              # Curated examples and candidate pools
├── experiments/
│   ├── run_parallel.py         # Main patching + optional automatic ablation driver
│   ├── run_patchings.py        # Patching implementations and CLI
│   ├── run_ablation_parallel.py# Standalone ablation for completed patching runs
│   ├── run_ablations.py        # Sequential ablation runner
│   ├── run_k_pipeline.py       # Resume-safe k-sensitivity workflow
│   ├── rescore_k_sweep.py      # Re-score saved token-level results at different k
│   ├── run_queue.py            # Run a queue of experiments
│   └── report_statistics.py    # Statistical summaries
├── src/
│   ├── models.py               # Model registry
│   ├── tasks.py                # Dataset/task registry and answer parsing
│   ├── templates.py            # Matched CoT and No-CoT prompt templates
│   ├── patching.py             # Activation-patching hooks
│   ├── patching_pipelines.py   # Token-wise scoring and joint patching
│   ├── ablation.py             # Zero-ablation implementation
│   ├── metrics.py              # JSD and MRR calculations
│   ├── analysis.py             # Statistical analysis
│   └── visualization.py        # Heatmap plotting
├── prepare_dataset.py          # Build curated datasets and candidate pools
└── requirements.txt
\`\`\`

## Installation

Use Python 3.10 or newer. A CUDA-capable GPU is recommended for the full activation-patching sweeps.

\`\`\`bash
git clone https://github.com/SrknOzturk/CoT_Mechanistic_Interpretability.git
cd CoT_Mechanistic_Interpretability

python -m venv .venv
\`\`\`

Activate the environment, then install dependencies:

**Linux/macOS**
\`\`\`bash
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
\`\`\`

**Windows PowerShell**
\`\`\`powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
\`\`\`

The Llama-3.2-1B checkpoint is gated. Accept its Hugging Face access terms and authenticate before running it:

\`\`\`bash
hf auth login
\`\`\`

## Prepare the datasets

The repository includes the source files and prebuilt BIG-bench source splits used by the preparation script. Run:

\`\`\`bash
python prepare_dataset.py
\`\`\`

To prepare only one task:

\`\`\`bash
python prepare_dataset.py --only svamp
python prepare_dataset.py --only bigbench_boolean_expressions
python prepare_dataset.py --only bigbench_web_of_lies
\`\`\`

The script writes the curated data and candidate pools under \`data/processed/\`. Candidate pools are larger than the target sample count so examples whose CoT generation never reaches the answer anchor can be replaced.

## Run the main experiments

A basic run for one model and one task:

\`\`\`bash
python experiments/run_parallel.py \\
  --model qwen2.5-0.5b \\
  --dataset svamp \\
  --target-n 64 \\
  --workers 1
\`\`\`

Use the following dataset keys for the main paper:

\`\`\`text
svamp
bigbench_boolean_expressions
bigbench_web_of_lies
\`\`\`

Repeat the command for each model–dataset combination. The available model keys are \`qwen2.5-0.5b\`, \`olmo2-1b\`, and \`llama3.2-1b\`.

By default, \`run_parallel.py\` runs both the standard Sequential Activation Patching experiment (\`normal\`) and the Random Activation Patching control (\`random\`). It then runs zero-ablation for the selected heads unless \`--no-ablation\` is specified. The random-activation run uses the matching standard-run outputs, so run the default workflow or complete the \`normal\` run before requesting \`random\` by itself.

Useful options:

| Option | Purpose |
|---|---|
| \`--target-n 64\` | Target number of successfully patched examples; skipped examples can be replaced from the reserve pool. |
| \`--n N\` | Restrict the candidate pool to its first \`N\` rows before selecting the target and reserve examples. |
| \`--heads-per-pos 3\` | Number of candidates retained per POS category; the main paper uses \`k = 3\`. |
| \`--global-top-k K\` | Optionally cap the final joint intervention to the best \`K\` distinct selected units. |
| \`--no-ablation\` | Run patching without the automatic CoT/No-CoT zero-ablation stage. |
| \`--workers N\` | Number of worker processes. Start with \`1\` if GPU memory is limited. |
| \`--gpu-total-gb GB\` | Supply total GPU memory when automatic worker sizing cannot detect it. |
| \`--max-steps N\` | Maximum CoT token positions scanned per example. |
| \`--ablation-max-new-tokens N\` | Generation budget for ablation runs. |
| \`--fresh\` | Discard checkpoints for the requested run and start it from scratch. |
| \`--dry-run\` | Print the execution plan without loading the model or spawning workers. |

The runner checkpoints progress by example and resumes when the same command is rerun. If a worker encounters CUDA out-of-memory, the parallel runner retries pending work with fewer workers.

### Outputs

Patching outputs are written to \`results/\`. The default combined runs produce separate files for each metric, for example:

\`\`\`text
results/
  qwen2.5-0.5b__svamp__step_by_step__normal__margin.json
  qwen2.5-0.5b__svamp__step_by_step__normal__jsd.json
  qwen2.5-0.5b__svamp__step_by_step__random__margin.json
  qwen2.5-0.5b__svamp__step_by_step__random__jsd.json
  ablation/
    ...
\`\`\`

The \`margin\` suffix corresponds to MRR in the paper. Ablation CSVs and checkpoints are stored under \`results/ablation/\` and \`results/checkpoints/\`, respectively.

## Additional analyses

### k-sensitivity analysis

The k-sweep reuses saved token-level results to evaluate different POS selection budgets without repeating the full localization sweep.

\`\`\`bash
python experiments/run_k_pipeline.py prepare \\
  --model olmo2-1b --dataset svamp --workers 1 --gpu-total-gb 48

python experiments/run_k_pipeline.py complete \\
  --model olmo2-1b --dataset svamp \\
  --margin-k 6 --jsd-k 15 --workers 1 --gpu-total-gb 48
\`\`\`

Choose \`--margin-k\` and \`--jsd-k\` based on the k-sweep results before running the \`complete\` stage.

### MLP-level interventions

The code also supports an experimental MLP-level variant, treating each layer's MLP output as one intervention unit:

\`\`\`bash
python experiments/run_parallel.py \\
  --model qwen2.5-0.5b --dataset svamp \\
  --target-n 64 --workers 1 --component mlp --heads-per-pos 3
\`\`\`

MLP-level results are stored separately from attention-head results. This extension should be treated as an additional experiment, not as one of the main attention-head results reported in the paper.

## Reproducibility notes

- Use the same model, dataset, template, target count, generation budget, and decoding options when comparing standard and random-activation runs.
- Some base-model generations may fail to reach the final-answer anchor within the token budget. These are recorded as skipped; the runner can draw replacements from the candidate reserve pool.
- \`--repetition-penalty\` and \`--no-repeat-ngram-size\` are available for tasks where base models repeat text instead of reaching the answer anchor. They are off by default.
- The repository contains additional task and exploratory experiment code beyond the main paper workflow. The commands above describe the primary three-model, three-task evaluation.
