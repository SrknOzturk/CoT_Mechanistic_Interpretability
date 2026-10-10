# SVAMP–Qwen cross-example patching and ablation

This experiment is intentionally fixed to the requested main configuration:

- model: `qwen2.5-0.5b`
- dataset: `svamp`
- prompt: `step_by_step`
- target questions: 32
- head-selection budget: `k=3` per POS category
- metrics: MRR and JSD, scored from the same head sweep
- ablation conditions: CoT and No-CoT
- ablation controls: unablated and equal-count random heads

For every target question B, a distinct donor question A supplies its generated
CoT trajectory and attention-head activations. The activations are inserted at
the final-answer position of B's No-CoT prompt. MRR and JSD use B's own clean
CoT logits as the target reference and B's No-CoT logits as the baseline.

After both cross-patching result files are complete, the same command reads the
heads selected separately by MRR and JSD and zero-ablates them throughout B's
generation trajectory. Both CoT and No-CoT prompting are evaluated. Each
condition records the unablated output, selected-head ablation, and an
equal-count random-head ablation control.

The 32 questions are sampled evenly from the eight `OperationCount × Type`
strata (four per stratum). Donors are a one-to-one permutation of those same
32 questions, are drawn from the same stratum as the target, and must have a
different gold answer. This means the run uses exactly 32 unique questions,
not 32 targets plus a separate donor pool.

Validate and inspect the pairing without loading the model:

```bash
python experiments/run_cross_svamp_qwen.py --plan-only
```

Run the complete GPU experiment (cross-patching followed automatically by
ablation):

```bash
python experiments/run_cross_svamp_qwen.py --device cuda
```

The default output directory is `results/cross/`. It contains the immutable
pair manifest, separate MRR and JSD patching JSON files, CoT and No-CoT
ablation CSV files for each selection metric, an ablation summary CSV, and
append-only JSONL checkpoints. Re-running the same command resumes completed
target IDs from the patching and ablation checkpoints.
