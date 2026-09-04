"""
src/utils.py
This module contains core utility functions for model loading, text generation, 
NER tagging, logit/entropy calculations, and text processing/ablation helpers.
"""

# 1. Standard Python Libraries
import os
import re
import random
from collections import defaultdict
from typing import Optional, Tuple, Callable, Union, List, Dict, Any

# 2. Third-Party Data & ML Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import nltk

# answer parsing/comparison is task-specific and lives in src.tasks;
# re-exported here so existing `from src.utils import ...` callers keep working
from src.tasks import AnswerTriggerNotFound, extract_last_number, get_task, numeric_equal
from src.models import MODELS
# prompt construction is template logic, not model logic; re-exported here
# so existing `from src.utils import make_*_prompt_from_row` keeps working
from src.templates import (
    get_template,
    make_cot_prompt_from_row,
    make_nocot_prompt_from_row,
)

# ===========================================================================
# 3. NLP Tools (Safe loading for SpaCy & NLTK)
# ===========================================================================

# Optional: Use spaCy for NER if installed (yields better results than NLTK)
_spacy_ner = None
try:
    import spacy
    _spacy_ner = spacy.load("en_core_web_sm")
except Exception:
    pass

# NLTK POS tagger (Penn Treebank); download once if missing
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
# Data required for NLTK NER (ne_chunk)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
# ===========================================================================

# ---------------------------------------------------------------------------
# Compatibility patch for newer transformers versions
# ---------------------------------------------------------------------------
if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    try:
        from transformers.utils import TRANSFORMERS_CACHE as _TRANSFORMERS_CACHE
        transformers.TRANSFORMERS_CACHE = _TRANSFORMERS_CACHE
    except Exception:
        default_cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        transformers.TRANSFORMERS_CACHE = default_cache_dir


# ===========================================================================
# Model Loading
# ===========================================================================
def load_model(model_name="qwen2.5-0.5b", device="cuda"):
    """
    Loads a pretrained Transformer-Lens model and returns it in evaluation mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # CLI commands use the short, stable keys from src.models. TransformerLens
    # expects its official model/repository name instead.
    resolved_name = MODELS[model_name].tl_name if model_name in MODELS else model_name
    print(f"Loading model: {model_name} ({resolved_name}) on {device}...")
    model = HookedTransformer.from_pretrained(
        resolved_name,
        device=device,
        dtype=torch.float32
    )
    model.eval()
    print(f"Model loaded successfully. Total layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
    return model


# ===========================================================================
# Data / Prompt / Answer Helpers
# ===========================================================================


def get_example_id_from_row(row: pd.Series, idx=None) -> str:
    """Safely extracts the ID from a dataframe row."""
    for col in ["example_id", "ID", "id"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return str(idx)


def get_answer_from_row(row: pd.Series) -> float:
    """Safely extracts the ground-truth numerical answer from a dataframe row."""
    for col in ["Answer", "answer", "true_answer", "target", "label"]:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    raise KeyError("Could not find an Answer/target column in the dataframe.")


def get_type_from_row(row: pd.Series) -> Optional[str]:
    """Safely extracts the question type from a dataframe row."""
    for col in ["Type", "type"]:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


def safe_accuracy(df_result: pd.DataFrame, column_name: str) -> float:
    """
    Calculates accuracy from a boolean column.
    Returns 0.0 if the column is missing or the dataframe is empty.
    """
    if df_result is None or len(df_result) == 0:
        return 0.0

    if column_name not in df_result.columns:
        return 0.0

    return df_result[column_name].fillna(False).mean() * 100


# ===========================================================================
# Token-by-Token Generation Helpers
# ===========================================================================

def _decode(model: HookedTransformer, tokens: torch.Tensor) -> str:
    """Safely converts TransformerLens to_string output to a string."""
    text = model.to_string(tokens)
    if isinstance(text, list):
        return text[0]
    return text


def _decode_generated_only(model: HookedTransformer, input_tokens: torch.Tensor, output_tokens: torch.Tensor) -> str:
    """Returns only the generated text (excluding the initial prompt)."""
    generated_tokens = output_tokens[0, input_tokens.shape[-1]:]
    return _decode(model, generated_tokens)


def _decode_single_token(model: HookedTransformer, token: torch.Tensor) -> str:
    """Safely decodes a single token tensor into a string."""
    text = model.to_string(token)
    if isinstance(text, list):
        return text[0]
    return text


def _append_token(output_tokens: torch.Tensor, next_token: torch.Tensor) -> torch.Tensor:
    """Appends a new token tensor to the existing sequence."""
    return torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)


def _is_numeric_answer_token(token_str: str) -> bool:
    """
    Determines if a token is a valid part of a numerical answer.
    Accepts: "38", " 38", ".", "0", "-", ","
    Rejects: " Therefore", " apples", "\n", " "
    """
    if token_str is None or token_str == "":
        return False

    stripped = token_str.strip()
    if stripped == "":
        return False

    allowed_chars = set("0123456789.,-")
    return all(ch in allowed_chars for ch in stripped)


def _strip_bos(model, text: str) -> str:
    """
    Removes the leading BOS marker from a decoded sequence.

    The original code sliced [13:], which is len("<|endoftext|>") -- Qwen's BOS
    rendering. That length is tokenizer-specific: Llama renders BOS as
    "<|begin_of_text|>" (17 characters), so a fixed slice would leave "ext|>"
    glued to the front of the string and silently corrupt every downstream
    comparison without raising an error.
    """
    tokenizer = getattr(model, "tokenizer", None)
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text

# ===========================================================================
# Old Standard Generation Blocks (Required by Patching)
# ===========================================================================
def generate_till_answer(model, prompt: str, max_new_tokens: int = 1024, task=None):
    """
    Generates text until the model produces a specific answer trigger and continues
    capturing the numerical result while handling multi-token digits.
    """
    task = task or get_task()
    device = next(model.parameters()).device
    model.reset_hooks()

    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    output_tokens = tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

            current_text = model.to_string(output_tokens)[0]
            if task.ends_reasoning(current_text):
                break

        answer_without_number = _strip_bos(model, model.to_string(output_tokens)[0])

        while True:
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            token_str = model.to_string(next_token)[0]

            if not task.is_answer_continuation(token_str):
                break
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

    final_text = _strip_bos(model, model.to_string(output_tokens)[0])
    return final_text, answer_without_number


def generate_full_answer_and_get_logits(model, prompt: str, max_new_tokens: int = 1024, task=None):
    """
    Generates the full answer using a CoT prompt and returns the logits of the first 
    token immediately following 'The answer is '.
    """
    task = task or get_task()
    device = next(model.parameters()).device
    model.reset_hooks()

    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    output_tokens = tokens.clone()

    with torch.no_grad():
        answer_token_logits = None
        for _ in range(max_new_tokens):
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

            current_text = model.to_string(output_tokens)[0]
            if task.ends_reasoning(current_text):
                logits = model(output_tokens[:, -2048:])
                answer_token_logits = logits[0, -1, :].clone()
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)
                break


        while True:
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            token_str = model.to_string(next_token)[0]

            if not task.is_answer_continuation(token_str):
                break
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

    if answer_token_logits is None:
        # the model never committed to an answer within the budget; the caller
        # decides whether to skip this example or treat it as fatal
        tail = model.to_string(output_tokens)[0][-120:]
        raise AnswerTriggerNotFound("CoT prompt", max_new_tokens, tail)

    full_text = model.to_string(output_tokens)[0]
    prompt_text = model.to_string(tokens)[0]
    if full_text.startswith(prompt_text):
        full_answer_text = full_text[len(prompt_text):]
    else:
        full_answer_text = _strip_bos(model, full_text)

    return full_answer_text, answer_token_logits


# ===========================================================================
# NLP & NER Utilities
# ===========================================================================
def _ner_label_for_last_word(context_str, context_max_words=80):
    """
    Returns the NER label of the last word. 
    Uses spaCy if available, otherwise falls back to NLTK ne_chunk.
    """
    if not context_str or not context_str.strip():
        return "O"
        
    if _spacy_ner is not None:
        try:
            doc = _spacy_ner(context_str[-2000:]) 
            if doc and len(doc):
                last_ent = doc[-1].ent_type_
                return last_ent if last_ent else "O"
        except Exception:
            pass
            
    try:
        all_words = nltk.word_tokenize(context_str)
        words = all_words[-context_max_words:] if len(all_words) > context_max_words else all_words
    except Exception:
        return "O"
        
    if not words:
        return "O"
        
    tagged = nltk.pos_tag(words)
    try:
        tree = nltk.ne_chunk(tagged, binary=False)
    except Exception:
        return "O"
        
    def leaves_with_parent_label(t, parent_ne="O"):
        if isinstance(t, nltk.tree.Tree):
            ne = t.label() if (hasattr(t, "label") and t.label() != "S") else parent_ne
            for c in t:
                yield from leaves_with_parent_label(c, ne)
        else:
            yield (t[0], t[1], parent_ne)
            
    try:
        per_word = list(leaves_with_parent_label(tree))
    except Exception:
        return "O"
        
    if not per_word:
        return "O"
    return per_word[-1][2]


# ===========================================================================
# Logits, Entropy & Patching Helpers
# ===========================================================================
def entropy_from_logits(logits):
    """Calculates entropy from the last position logits: H = -sum(p*log(p))."""
    if logits.dim() > 1:
        logits = logits.reshape(-1)
    p = F.softmax(logits, dim=-1)
    p = p.clamp(min=1e-10)
    return -(p * p.log()).sum().item()


def print_top10_with_logits_and_entropy(logits_1d, model, title, k=10):
    """Prints top-k tokens: prob, logit, token; along with distribution entropy."""
    probs = F.softmax(logits_1d, dim=-1)
    ent = entropy_from_logits(logits_1d)
    top_p, top_i = torch.topk(probs, k=k, dim=-1)
    
    print(title)
    print(f"Entropy: {ent:.6f}")
    print("-" * 80)
    for i, (p, idx) in enumerate(zip(top_p, top_i), 1):
        idx_val = idx.item()
        logit_val = logits_1d[idx_val].item()
        tt = torch.tensor([[idx_val]], device=logits_1d.device)
        token_str = model.to_string(tt)[0].strip()
        print(f"{i:2d}. Token ID: {idx_val:6d} | Prob: {p.item():.6f} | Logit: {logit_val:8.4f} | Token: '{token_str}'")
    print("-" * 80)


def _merge_top_heads_for_pos_margin_ratio(
    existing: list,
    heatmap: torch.Tensor,
    clean_cache,
    n_heads: int,
    k: int,
) -> list:
    """Retrieves the top-k cells from the heatmap at each step; preserves the highest score and vector per (layer, head)."""
    flat = heatmap.flatten()
    kk = min(k, int(flat.numel()))
    if kk == 0:
        return existing
        
    vals, idxs = torch.topk(flat, k=kk)
    by_key = {}
    for item in existing:
        by_key[(item["layer"], item["head"])] = item
        
    for val, lin_idx in zip(vals, idxs):
        score = float(val.item())
        li = int(lin_idx.item())
        layer = li // n_heads
        head = li % n_heads
        key = (layer, head)
        clean_z = clean_cache["z", layer]
        vec = clean_z[0, -1, head, :].detach().cpu().clone()
        cand = {"score": score, "layer": layer, "head": head, "vec": vec}
        
        if key not in by_key or score > by_key[key]["score"]:
            by_key[key] = cand
            
    return sorted(by_key.values(), key=lambda x: -x["score"])[:k]


def _merge_top_heads_for_pos_jsd(prev_entries, hm, clean_cache, nh, k):
    """
    Updated Top-K Head merging function specifically for the JSD metric.
    Unlike Margin scores, the lowest score (closest to 0) is the best for JSD.
    """
    flat_hm = hm.flatten()
    current_k = min(k, flat_hm.numel())
    
    topk_vals, topk_indices = torch.topk(flat_hm, current_k, largest=False)
    
    current_entries = []
    for val, idx in zip(topk_vals, topk_indices):
        l, h = divmod(idx.item(), nh)
        hook_name = f"blocks.{l}.attn.hook_z"
        vec = clean_cache[hook_name][0, -1, h, :]
        
        current_entries.append({
            "layer": l,
            "head": h,
            "score": float(val.item()),
            "vec": vec
        })
        
    combined = prev_entries + current_entries
    
    best_unique = {}
    for entry in combined:
        key = (entry["layer"], entry["head"])
        if key not in best_unique or entry["score"] < best_unique[key]["score"]:
            best_unique[key] = entry
            
    final_list = list(best_unique.values())
    final_list.sort(key=lambda x: x["score"])
    
    return final_list[:k]

import re
import random
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


# ============================================================
# 1. DATA / PROMPT / ANSWER HELPERS
# ============================================================


# ============================================================
# 2. HEAD SELECTION / ABLATION HELPERS
# ============================================================

def build_heads_by_example_id_from_curated(curated_normal_results, selected_heads_key="selected_heads"):
    """
    curated_normal_results formatı:

    [
        {
            "example_id": "...",
            "selected_heads": [
                {"layer": 3, "head": 5},
                {"layer": 7, "head": 2},
                ...
            ]
        },
        ...
    ]

    Çıktı:
        {
            example_id: [(layer, head), ...]
        }
    """
    heads_by_example_id = {}

    for item in curated_normal_results:
        example_id = str(item["example_id"])
        selected_heads = item.get(selected_heads_key, [])

        heads = []
        for h in selected_heads:
            if h is None:
                continue

            layer = int(h["layer"])
            head = int(h["head"])
            heads.append((layer, head))

        heads_by_example_id[example_id] = heads

    return heads_by_example_id


def group_heads_by_layer(heads):
    layer_dict = defaultdict(list)

    for layer, head in heads:
        layer_dict[int(layer)].append(int(head))

    return layer_dict


def make_zero_ablation_hooks(heads):
    """
    Verilen [(layer, head), ...] listesindeki attention head'lerin
    hook_z aktivasyonlarını sıfırlar.

    Ablate edilen şey:
        blocks.{layer}.attn.hook_z içindeki ilgili head output'u.

    z shape:
        [batch, position, n_heads, d_head]

    Yapılan işlem:
        z[:, :, head, :] = 0.0
    """
    if not heads:
        return []

    layer_dict = group_heads_by_layer(heads)
    hooks = []

    for layer, head_list in layer_dict.items():
        hook_name = f"blocks.{layer}.attn.hook_z"

        def hook_fn(z, hook, head_list=head_list):
            for h in head_list:
                z[:, :, h, :] = 0.0
            return z

        hooks.append((hook_name, hook_fn))

    return hooks


def sample_random_heads_matched_layers_for_example(model, selected_heads_list, seed=42):
    """
    Direct Equation için layer-matched random control.

    Selected heads hangi layer'larda kaç tane ise,
    random heads de aynı layer'lardan aynı sayıda seçilir.
    """
    random.seed(seed)

    n_heads = model.cfg.n_heads
    layer_counts = defaultdict(int)

    for layer, head in selected_heads_list:
        layer_counts[int(layer)] += 1

    random_heads = []

    for layer, count in layer_counts.items():
        available_heads = list(range(n_heads))
        chosen = random.sample(available_heads, count)

        for h in chosen:
            random_heads.append((layer, h))

    return random_heads


def sample_random_heads_same_count_for_example(model, num_heads, seed=42):
    """
    No-CoT ve CoT için aynı sayıda random head control.
    Layer matching yapmaz, bütün layer-head havuzundan seçer.
    """
    random.seed(seed)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]

    if num_heads > len(all_heads):
        num_heads = len(all_heads)

    return random.sample(all_heads, num_heads)


