# Mechanistic Interpretability Framework: Attention Head Patching & Ablation

This framework provides a modular implementation for investigating the internal decision-making processes of Large Language Models (LLMs). By utilizing **Activation Patching** and **Zero-Ablation** techniques, this project allows researchers to isolate specific attention heads and quantify their causal contribution to the model's reasoning capabilities, particularly on mathematical word problem datasets like **SVAMP**.

## 🚀 Overview

Mechanistic interpretability aims to reverse-engineer the "black box" of neural networks. This repository automates the identification and validation of critical attention heads:

* **Patching:** Probes which attention heads in the model carry critical information for a specific task (e.g., CoT reasoning) by injecting activations from clean prompts into corrupted ones.
* **Ablation:** Validates the causal necessity of these heads by zeroing out their activations (zero-ablation) and measuring the impact on model performance.

## 📂 Project Structure

```text
.
├── data/               # Raw and processed datasets (e.g., SVAMP)
├── experiments/        # Execution scripts for experiments
├── results/            # JSON/CSV outputs from patching and ablation runs
├── src/                # Core implementation logic:
│   ├── ablation.py     # Ablation hooks and control experiment logic
│   ├── data_loader.py  # Dataset handling
│   ├── metrics.py      # JSD and Margin Recovery Ratio factories
│   ├── patching.py     # Activation patching primitives
│   ├── pipelines.py    # Orchestration pipelines for patching sweeps
│   └── utils.py        # General helpers (NER, POS, generation, decoding)
├── requirements.txt    # Project dependencies
└── README.md

```


## 📊 Workflow

To perform a complete interpretability analysis, execute the following steps in order:

### 1. Patching (Exploration)

Identify the heads that are most important for the model's output distribution. This generates the "importance map" needed for ablation.

```bash
python experiments/run_patching.py

```

### 2. Ablation (Validation)

Test the causal necessity of the identified heads by zeroing them out and measuring performance impact (Accuracy/Recovery Score).

```bash
python experiments/run_ablation.py

```

## 🔍 Methodology Highlights

* **Metric Factories:** Includes flexible factories for Margin Recovery Ratio (MRR) and Jensen-Shannon Divergence (JSD).
* **Cross-Patching:** Supports experiments where Chain-of-Thought (CoT) traces are patched across different problem instances.
* **Control Experiments:** Implements random head ablation to serve as a baseline for statistical significance.

