# Mechanistic Interpretability of Chain-of-Thought Reasoning in Math Word Problems

This repository contains the official implementation for the paper: **"Mechanistic Interpretability of Chain-of-Thought Reasoning in Math Word Problems: CoT as a Meta-Controller"**.

## 🚀 Overview
In this work, we hypothesize that Chain-of-Thought (CoT) prompting acts as a **meta-controller** within the network, dynamically coordinating and activating specific problem-solving sub-circuits. We introduce **Sequential Multi-Head Patching**, a novel interpretability method that maps these functional circuits by analyzing activation flow across multiple tokens.

## 📂 Repository Structure
```text
.
├── data/               # SVAMP dataset and processed subsets
├── experiments/        # Scripts for running patching and ablation experiments
├── results/            # JSON/CSV outputs from experiment runs
├── src/                # Core implementation logic:
│   ├── ablation.py     # Zero-ablation hooks and control experiment logic
│   ├── data_loader.py  # Dataset handling and pre-processing
│   ├── metrics.py      # JSD and Margin Recovery Ratio factories
│   ├── patching.py     # Activation patching primitives
│   ├── pipelines.py    # Orchestration pipelines for experiments
│   └── utils.py        # General helpers (NER, POS, generation, decoding)
├── requirements.txt    # Project dependencies
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



## 📊 Workflow

To replicate the study's findings, execute the following steps:

### 1. Patching (Exploration)

Identify the attention heads most critical to the model's output distribution using sequential patching.

```bash
python experiments/run_patching.py

```

### 2. Ablation (Validation)

Validate the causal necessity of the identified heads by zeroing out their activations (zero-ablation) and measuring the performance impact.

```bash
python experiments/run_ablation.py

```

## 🔍 Methodology

* **Sequential Multi-Head Patching:** Unlike naive single-token intervention, our approach tracks the causal influence of heads across the full reasoning trace.
* **Meta-Controller Hypothesis:** We investigate how CoT tokens coordinate activation flow to calculation and abstraction sub-circuits.

