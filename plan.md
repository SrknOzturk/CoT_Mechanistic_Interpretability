# CoT Meta-Controller — ARR October Cycle Plan

Working document for the resubmission of *"Mechanistic Interpretability of
Chain-of-Thought Reasoning in Math Word Problems: CoT as a Meta-Controller"*
(EMNLP 2026 → Findings; resubmitting to ACL ARR, October cycle).

Last updated: 2026-08-29

---

## 1. Why this round of work

Three reviewers and the AC converged on the same criticism: **the evaluation is
too narrow**. One model (Qwen2.5-0.5B), one dataset (SVAMP, 32 examples), one
prompt structure. Secondary asks were statistical uncertainty, sensitivity to
the head-selection and POS-grouping choices, and documented compute cost.

| Source | Ask | How it is addressed |
|---|---|---|
| vyFV, Gcgt | more models | Qwen2.5-0.5B + OLMo-2-1B + Llama-3.2-1B (base) |
| Gcgt | more datasets, "why not PrOntoQA?" | SVAMP + ProntoQA (BBH pending) |
| Gcgt | vary the prompt structure | template registry; `step_by_step` is the new primary |
| AC | statistical uncertainty | bootstrap CIs, paired tests, effect sizes |
| AC | sensitivity to k / POS grouping | offline k-sweep + head-count-matched control |
| aPyH | computational overhead | telemetry + a documented cost model |
| vyFV | explain the patching site precisely | single anchor, asserted invariants |

---

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | ProntoQA re-sourced as **True/False QA** | The downloaded OOD dumps were the proof-generation variant: no `answer` field at all |
| D2 | Llama = `meta-llama/Llama-3.2-1B` **base**, not Instruct | An instruction-tuned model reasons even under the No-CoT prompt, collapsing the contrast the paper depends on |
| D3 | **All models float32** | Removes precision as a confound between models |
| D4 | Parallelism is **process-level**: N copies of one model, one example each | Batching examples would need left-padding, which moves the patch site off the true last token |
| D5 | Matrix = 3 models × 2 datasets × 1 template | BBH deferred pending subtask selection |
| D6 | Template = `Q: … A: Let's think step by step. … Q: … A: Let's think step by step.` | A new model is not guaranteed to produce CoT without an explicit cue |
| D7 | **n = 64** per dataset (pools built at ~1.5x, 96, for pre-filter slack) | Cuts compute further while keeping the design's balance |
| D8 | Experiments = **normal sequential patching + random control**. Cross-patching dropped | See §3 |
| D9 | **Margin and JSD scored in one scan** | They read the same patched logits; two scans paid for every forward pass twice |
| D10 | Ablation kept (No-CoT + CoT). **Equation condition dropped** | Patching is the contribution; ablation is verification |
| D11 | Generation limits = 1024; the patching loop has **no fixed cap** — it runs the pre-filter's measured trace length | Guarantees no truncated trace without the runaway cost of a blind cap |
| D12 | Examples whose trace never reaches the anchor are **skipped**; the kept set is the **intersection across all three models** | Skipping is model-dependent, and cross-model claims need paired comparison on common examples |
| D13 | Statistics: bootstrap CIs, paired test vs random, k-sweep, POS ablation with matched control | First four cost ~0 GPU |
| D14 | An example whose trace never reaches the answer trigger is **skipped, not crashed on** — recorded with `skipped=True` and a `skip_reason`, excluded from every statistic | A base model failing to produce a well-formed trace is an expected outcome, and losing the whole run's progress to one bad example was the alternative |

### Prompt structure

```
CoT   : Q: {demo} A: Let's think step by step. {reasoning} The answer is {a}.
        Q: {target} A: Let's think step by step.

No-CoT: Q: {demo} A: The answer is {a}.
        Q: {target} A:                            + " The answer is "
```

The cue appears **only** on the CoT side. That asymmetry is the manipulation.

---

## 3. Finding that changed the design

Re-analysing the existing Qwen/SVAMP results (no GPU) showed **cross-patching and
normal patching are statistically indistinguishable**:

| Metric | normal | cross | difference, 95% CI | p | effect size |
|---|---|---|---|---|---|
| Final JSD | 0.2415 | 0.2594 | [−0.081, +0.043] | 0.875 | −0.03 |
| Prob. increase | 0.2506 | 0.2075 | [−0.042, +0.140] | 0.818 | +0.05 |
| Logit increase | 1.3029 | 1.1300 | [−0.355, +0.934] | 0.832 | +0.05 |

And 24 of 32 donors came from a *different* arithmetic operation type, yet still
scored 0.2688 [0.199, 0.337] — overlapping the same-type donors (0.2311). Heads
found from a subtraction problem recover an addition problem's answer equally well.

Meanwhile the core claim is solid:

| Condition | Final JSD [95% CI] |
|---|---|
| Unpatched baseline | 0.3847 [0.3295, 0.4354] |
| **Normal patching** | **0.2415 [0.1728, 0.3168]** |
| Cross patching | 0.2594 [0.1933, 0.3251] |
| Random patching | 0.4018 [0.3343, 0.4609] |

normal vs random: p < 1e-4, r = −0.795, favouring normal in 27/32 examples.

**Consequence for the paper:** the recovered signal is a question-general CoT
control signal, not question-specific content. That supports the meta-controller
framing more strongly than the current text claims, but the framing must be
adjusted — cross-patching cannot be presented as a discriminating control.

Two further observations: the random control is well-behaved (probability
increase −0.0025, CI spanning zero; its JSD is slightly *worse* than not
patching), and `logit_increase` is noisy (sd 3.86 under the random condition),
so JSD and `prob_increase` should carry the argument.

Head selection is concentrated, not diffuse: only 94 of 336 heads are ever
selected, and `L14.H10` appears in 21/32 examples — a direct answer to Gcgt's
"any mechanism looks distributed if you coarsen the granularity".

---

## 4. Bugs found and fixed

### Would have produced wrong results silently

| Where | Problem |
|---|---|
| `utils.py` `[13:]` | Looked like `len("The answer is")`; actually `len("<\|endoftext\|>")`, a BOS strip. Both are 13 on Qwen. **Llama's BOS is 17 characters**, so the slice left `ext\|>` glued to the prompt and corrupted every step from step 0 — with no error. Replaced by `_strip_bos`, verified per tokenizer |
| `make_nocot_prompt_from_row` | Read the column `"PromptWithoutCot"`, which no longer exists after templating → `KeyError` on ProntoQA, a completely different zero-shot prompt on SVAMP. **The ablation would have measured a different protocol than the patching it verifies.** Now template-aware; verified byte-identical to the patching prompt across all 204 rows |
| same function | `if "The answer is " in prompt` should be `endswith`: the 1-shot demo contains the trigger, so it was never appended to the target |
| `utils.py` | 14 functions defined twice (notebook paste); Python silently used the second. Normalised comparison showed all 14 pairs functionally identical, so removal changed nothing but the ambiguity |
| `ctx` | 1024 in the sequential runs, 2048 in the controls, `model.cfg.n_ctx` in the pipeline. Harmless on SVAMP (~120 tokens), silent truncation on ProntoQA (~520) |

### Blocked the repo from running at all

| Where | Problem |
|---|---|
| `run_patchings.py:34` | `from src.metrics import …` — module was named `patching_evaluation_metrics.py`. Renamed (the file's own docstring and the README already said `metrics.py`) |
| `ablation.py` | Imported six functions that existed nowhere in the repo or git history; recovered from `Multi_head_patching.ipynb` |
| `sys.path.append` | An unrelated `src` package in site-packages shadowed the project's. `append` → `insert(0)` |
| `run_ablations.py` | Wrong results filenames; outputs written to CWD instead of `results/` |

### Reproducibility

Cross-patching's donor draw and the random control's Gaussian vectors were both
unseeded — neither control could be reproduced. Both now derive a deterministic
per-example seed. `random.seed()` calls that mutated global RNG state replaced
with local `Random` instances.

### Skip handling (new)

`generate_full_answer_and_get_logits` used to raise a bare `ValueError` when the
trigger never appeared, crashing the whole run and losing every example already
processed. It now raises `AnswerTriggerNotFound` (`src/tasks.py`), which the
drivers catch:

- `run_patchings.py` records a `skipped_record()` — same shape as a real record,
  `metrics` left empty — and continues
- `run_ablations.py` flags a row `skipped` when the unablated run never reached
  the trigger, and excludes it from the accuracy computed in the summary
- `src/analysis.py`'s `metric_vector` / `align` drop skipped rows automatically,
  so nothing downstream needs to know skips exist
- `skip_summary()` reports the count and reason per example; both drivers and
  `report_statistics.py` print it

Verified end to end with a scripted model that never emits the trigger: the run
continues, the skip is recorded, and it does not reach `bootstrap_ci`.

### Data

`results/multi_head_patching_with_margin_results.json` and its cross counterpart
are **truncated JSON** and cannot be parsed. Root cause: ~30 MB of `hm_matrix`
serialised as JSON text. Not re-run, since everything is being re-run anyway;
the four intact files are kept as a refactor reference.

---

## 5. Architecture

```
src/tasks.py       answer parsing, comparison, answer-continuation gate  (per task)
src/templates.py   prompt structures + invariant checks                  (per template)
src/models.py      model registry with head geometry and size            (per model)
src/data_loader.py curation: raw -> curated -> balanced candidate pool
src/patching.py    the layer x head sweep; scores several metrics per pass
src/metrics.py     margin recovery ratio, JSD
src/ablation.py    zero-ablation generation + No-CoT / CoT pipelines
src/analysis.py    bootstrap CIs, paired tests, offline k-sweep
src/gpu_planning.py worker-count and cost model

prepare_dataset.py builds the curated tables and candidate pools
notebooks/         inspect_datasets.ipynb -- what the experiments load, with outputs saved
experiments/run_patchings.py    the patching driver
experiments/run_ablations.py    the ablation driver
experiments/report_statistics.py statistics report
experiments/plan_compute.py     worker count + GPU-hour estimate
```

The task decides how an answer is parsed and which tokens still belong to it;
the template decides how prompts are laid out. Nothing downstream hardcodes
either, so a new dataset is a registry entry rather than a new code path.

### Data pipeline

| Stage | SVAMP | ProntoQA |
|---|---|---|
| raw | `SVAMP.json` (1000) | `prontoqa/{2,3,4,5}hop_1shot_seed42.json` |
| curated | `svamp_curated.json` (1000) | `prontoqa_curated.json` (200) |
| **candidates** ← what runs | `svamp_candidates.json` (104) | `prontoqa_candidates.json` (100) |

Anything else in `data/processed/` is an intermediate and is read by nothing;
`notebooks/inspect_datasets.ipynb` labels each file with its role.

Pools are deliberately larger than 100 because the pre-filter and the
three-model intersection (D12) will shrink them.

ProntoQA is generated locally from `prontoqa-main`:

```
python run_experiment.py --model-name json --model-size dummy \
  --ontology fictional --num-trials 40 --few-shot-examples 1 \
  --min-hops 1 --max-hops 5 --hops-skip 1 --seed 42
```

**Without `--proofs-only`** — that flag is what produced the unusable
answer-less dumps. Result: 200 examples, 101 True / 99 False. Hops 2–5 × 25 are
kept; 1-hop is excluded as too short to scan.

---

## 6. Compute

### Why process-level parallelism

Batching examples would require left-padding unequal prompts, and the patch site
is defined as the last position — padding moves it. Running N independent
single-example processes avoids that completely. It pays off because at batch 1
a 0.5–1B model is dominated by kernel-launch overhead rather than arithmetic, so
concurrent processes fill each other's idle gaps.

| model | weights | cache | logits | runtime | total | workers on 48 GB |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 1.8G | 1.0G | 0.3G | 0.9G | 4.0G | 8 |
| OLMo-2-1B | 5.5G | 1.3G | 0.2G | 0.9G | 7.9G | 4 |
| Llama-3.2-1B | 4.6G | 1.5G | 0.2G | 0.9G | 7.3G | 5 |

Qwen fits ~12 copies by memory; the cap of 8 reflects compute saturation, not
VRAM, and should be replaced by a calibration measurement.

### Cost, n = 100

| model | dataset | steps | GPU-h |
|---|---|---|---|
| Qwen2.5-0.5B | SVAMP | 55 | 23.4 |
| Qwen2.5-0.5B | ProntoQA | 85 | 35.7 |
| OLMo-2-1B | SVAMP | 55 | 27.6 |
| OLMo-2-1B | ProntoQA | 85 | 42.2 |
| Llama-3.2-1B | SVAMP | 55 | 51.5 |
| Llama-3.2-1B | ProntoQA | 85 | 78.6 |
| | | **TOTAL** | **~259** |

**≈ 78 wall-clock hours ≈ 3.3 days** at 65% scaling efficiency
(`python experiments/plan_compute.py --n 64`).

This does not yet subtract skipped examples — the pre-filter will lower it further by however many candidates never reach the anchor.

### What the optimisations bought

| Configuration | GPU-h | days |
|---|---|---|
| No optimisation, with cross-patching | 1253 | 14.7 |
| + last-position slicing | 977 | 11.4 |
| + combined margin/JSD scan | 494 | 5.8 |
| + cross-patching dropped | **405** | **5.1** |

Two are free in the sense that output is unchanged:

- **Last-position slicing** — the metric only reads `logits[0, -1, :]`, but every
  pass computes logits for all positions. Hooking the last block's
  `hook_resid_post` to return `resid[:, -1:, :]` makes `ln_final` and `unembed`
  run on one position. Both are positionwise, so the result is identical. ~22%.
- **Combined scan** — margin and JSD read the same patched logits. Verified
  bit-for-bit identical to two separate scans, on a test where the two metrics
  genuinely rank heads differently. Exactly 2×.

### Uncertainty

Two inputs are estimates until measured:

- **Trace length** (assumed 55 / 85 steps) — the pre-filter measures this exactly
- **Scaling efficiency** (assumed 65%) — 45% → 8.4 days, 85% → 4.4 days; a ~1
  GPU-hour sweep at P ∈ {1,2,4,8} settles it

`python experiments/plan_compute.py --gpu-used 28 --efficiency 0.45` re-runs the
model under any assumption.

---

## 7. Status

### Done, verified without a GPU

| Area | State |
|---|---|
| Repo runs | 15 modules parse, every `src.*` import resolves, no duplicate or undefined names |
| Bugs | §4 fixed; BOS corruption and prompt-protocol mismatch each verified by test |
| `tasks` / `templates` / `models` | Written; 104/104 and 100/100 answer round-trip on real data |
| ProntoQA data | Generated, curated, balanced (2–5 hop × 25) |
| Drivers | `--model/--dataset/--template/--experiments`; namespaced outputs into `results/` |
| Dual-metric scan | Bit-for-bit equivalent to two separate scans |
| Ablation | Task-aware; hooks verified to wrap the whole trajectory; prompts byte-identical to patching |
| Statistics | `results/statistics/qwen_svamp_jsd_statistics.txt` |
| Dataset inspection | `notebooks/inspect_datasets.ipynb`, executed with outputs saved |
| Cost model | `experiments/plan_compute.py` |

### Next

1. **Pre-filter** — generate every candidate's trace on all three models, drop
   those that never reach the anchor, keep the intersection's first 100, record
   skip rates. First GPU job; also prices every cell exactly.
2. **Calibration sweep** — throughput at P ∈ {1,2,4,8} (~1 GPU-h).
3. **Telemetry** — `torch.cuda.Event`-synchronised timing per token step. The
   existing 29.58 s figure was measured without synchronisation.
4. **Last-position slicing** — implement behind an equivalence test.
5. **Production runs** — Qwen → OLMo → Llama, `normal` then `random_*`.
6. **Ablation runs** — driven by the primary results.
7. **Analysis** — k-sweep, POS ablation with matched control, compute appendix.

### Open

| # | Item | Blocks |
|---|---|---|
| O1 | SRL library for ProntoQA (advisor) | nothing — labels are re-derivable offline from saved heatmaps |
| O2 | BBH subtask selection (advisor) | the third dataset only |
| O3 | Whether to reframe cross-patching in the paper per §3 | writing, not code |

**Note on BBH:** avoid the multiple-choice subtasks. Their answers are `(B)`, so
the first token after the anchor is `(`, which carries no information about the
choice. `boolean_expressions`, `navigate`, `web_of_lies` and
`multistep_arithmetic_two` reuse the two answer handlers that already exist.

---

## 8. Running it

```bash
python prepare_dataset.py                    # build curated tables + candidate pools
jupyter lab notebooks/inspect_datasets.ipynb # inspect what the experiments load
python experiments/plan_compute.py --n 100   # workers and GPU-hour estimate

python experiments/run_patchings.py --model qwen2.5-0.5b --dataset svamp \
    --experiments normal random_margin random_jsd
python experiments/run_ablations.py --model qwen2.5-0.5b --dataset svamp
python experiments/report_statistics.py --results-dir results
```

`normal` writes `<model>__<dataset>__<template>__normal__margin.json` and
`…__jsd.json`. The random controls read those for their head counts, so they
must run after.
