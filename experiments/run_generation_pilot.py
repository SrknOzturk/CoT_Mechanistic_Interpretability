"""Compare decoding controls without paying for activation patching.

The pilot uses the exact greedy decoding and answer-trigger convention of the
main experiment, records every generated trace, and can be resumed from its
JSONL checkpoint.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models import MODELS  # noqa: E402
from src.tasks import TASKS, get_task  # noqa: E402
from src.templates import DEFAULT_TEMPLATE, TEMPLATES, get_template  # noqa: E402
from src.utils import DecodingConfig, apply_decoding_penalties, load_model  # noqa: E402


CONFIGS = (
    ("baseline", DecodingConfig(1.0, 0)),
    ("penalty_only", DecodingConfig(1.05, 0)),
    ("ngram_only", DecodingConfig(1.0, 4)),
    ("penalty_and_ngram", DecodingConfig(1.05, 4)),
)


def decode(model, tokens):
    text = model.to_string(tokens)
    return text[0] if isinstance(text, list) else text


def repetition_fraction(ids, n=4):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def generate(model, prompt, task, decoding, max_new_tokens, ctx):
    device = next(model.parameters()).device
    model.reset_hooks()
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    output_tokens = prompt_tokens.clone()
    generated_ids = []
    reached_trigger = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(output_tokens[:, -ctx:])
            step_logits = apply_decoding_penalties(
                logits[0, -1, :], generated_ids, decoding
            )
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)
            generated_ids.append(int(next_token.item()))

            if task.ends_reasoning(decode(model, output_tokens)):
                reached_trigger = True
                # Match the main experiment: answer tokens are greedy and do
                # not receive reasoning-phase repetition controls.
                logits = model(output_tokens[:, -ctx:])
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)
                generated_ids.append(int(next_token.item()))

                for _ in range(8):
                    logits = model(output_tokens[:, -ctx:])
                    next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                    token_str = decode(model, next_token)
                    if not task.is_answer_continuation(token_str):
                        break
                    output_tokens = torch.cat(
                        [output_tokens, next_token.unsqueeze(0)], dim=1
                    )
                    generated_ids.append(int(next_token.item()))
                break

    full_text = decode(model, output_tokens)
    prompt_text = decode(model, prompt_tokens)
    generated_text = (
        full_text[len(prompt_text):]
        if full_text.startswith(prompt_text)
        else full_text
    )
    prediction = task.extract(generated_text) if reached_trigger else None
    return {
        "completed": reached_trigger,
        "prediction": prediction,
        "generated_tokens": len(generated_ids),
        "fourgram_repetition_fraction": repetition_fraction(generated_ids),
        "generated_text": generated_text,
    }


def load_existing(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_summary(records, path):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)

    rows = []
    for mode, _ in CONFIGS:
        group = grouped.get(mode, [])
        if not group:
            continue
        completed = [r for r in group if r["completed"]]
        rows.append({
            "mode": mode,
            "n": len(group),
            "completed": len(completed),
            "completed_pct": 100 * len(completed) / len(group),
            "correct": sum(bool(r["correct"]) for r in group),
            "accuracy_pct": 100 * sum(bool(r["correct"]) for r in group) / len(group),
            "mean_generated_tokens": sum(r["generated_tokens"] for r in group) / len(group),
            "mean_fourgram_repetition_pct": 100 * sum(
                r["fourgram_repetition_fraction"] for r in group
            ) / len(group),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="olmo2-1b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="prontoqa", choices=sorted(TASKS))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATES))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results", "analysis"))
    args = ap.parse_args()

    task = get_task(args.dataset)
    template = get_template(args.template)
    data_path = os.path.join(REPO_ROOT, "data", "processed", task.dataset_file)
    df = pd.read_json(data_path).head(args.n)

    os.makedirs(args.results_dir, exist_ok=True)
    stem = f"{args.model}__{args.dataset}__{args.template}__generation_pilot_shared_demo"
    detail_path = os.path.join(args.results_dir, stem + ".jsonl")
    summary_path = os.path.join(args.results_dir, stem + "__summary.csv")
    records = load_existing(detail_path)
    done = {(str(r["example_id"]), r["mode"]) for r in records}

    print(f"Loading {args.model} on {args.device}...", flush=True)
    model = load_model(args.model, args.device)

    with open(detail_path, "a", encoding="utf-8") as out:
        for _, row in df.iterrows():
            example_id = str(row[task.id_column])
            prompt = str(row[template.cot_col])
            gold = task.gold_from_row(row)
            for mode, decoding in CONFIGS:
                if (example_id, mode) in done:
                    continue
                result = generate(
                    model, prompt, task, decoding, args.max_new_tokens, args.ctx
                )
                correct = bool(
                    result["completed"]
                    and task.answers_equal(result["prediction"], gold)
                )
                record = {
                    "example_id": example_id,
                    "mode": mode,
                    "repetition_penalty": decoding.repetition_penalty,
                    "no_repeat_ngram_size": decoding.no_repeat_ngram_size,
                    "gold_answer": gold,
                    "correct": correct,
                    **result,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                records.append(record)
                print(
                    f"{example_id:18s} {mode:18s} "
                    f"complete={result['completed']} correct={correct} "
                    f"tokens={result['generated_tokens']:3d} "
                    f"rep4={100 * result['fourgram_repetition_fraction']:.1f}%",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    rows = write_summary(records, summary_path)
    print("\nSummary")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nDetail:  {detail_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
