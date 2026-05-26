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
    print(f"Loading model: {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float32
    )
    model.eval()
    print(f"Model loaded successfully. Total layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
    return model


# ===========================================================================
# Data / Prompt / Answer Helpers
# ===========================================================================

def clean_equation_to_equals_format(equation: str) -> str:
    """
    Cleans an equation string, removes parentheses, normalizes spacing,
    and appends an equals sign.
    """
    equation = str(equation).strip()
    equation = equation.replace("(", "").replace(")", "")
    equation = re.sub(r"\s+", " ", equation).strip()
    return equation + " = "


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


def make_nocot_prompt_from_row(row: pd.Series) -> str:
    """
    Constructs a No-CoT prompt. Appends 'The answer is ' if missing.
    """
    for col in ["PromptWithoutCot", "prompt_no_cot"]:
        if col in row.index and pd.notna(row[col]):
            prompt = str(row[col])
            if "The answer is " in prompt:
                return prompt
            return prompt.rstrip() + " The answer is "

    q = row["Question"] if "Question" in row.index else row["question"]
    return f"{q} The answer is "


def make_cot_prompt_from_row(row: pd.Series) -> str:
    """
    Constructs a CoT prompt. Appends 'Let's think step by step.' if missing.
    """
    for col in ["PromptWithCot", "prompt_cot"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])

    q = row["Question"] if "Question" in row.index else row["question"]
    return f"{q} Let's think step by step."


def make_direct_equation_prompt_from_row(row: pd.Series) -> Optional[str]:
    """
    Constructs a Direct Equation prompt.
    Example: '76.0 - 25.0 = The answer is '
    """
    if "Equation" not in row or pd.isna(row["Equation"]):
        return None

    equation_str = clean_equation_to_equals_format(row["Equation"])
    return equation_str + "The answer is "


def extract_last_number(text: str) -> Optional[float]:
    """Extracts the final numeric value from a generated text block."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    if matches:
        return float(matches[-1])
    return None


def numeric_equal(pred, gold, tol=1e-4) -> bool:
    """Safely compares two numeric values with a tolerance buffer."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < tol
    except Exception:
        return False


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


# ===========================================================================
# Old Standard Generation Blocks (Required by Patching)
# ===========================================================================
def generate_till_answer(model, prompt: str, max_new_tokens: int = 500):
    """
    Generates text until the model produces a specific answer trigger and continues
    capturing the numerical result while handling multi-token digits.
    """
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
            if current_text.endswith("The answer is ") or current_text.endswith("The answer is -"):
                break

        answer_without_number = model.to_string(output_tokens)[0][13:]
        non_problematic_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", ",", "-"]

        while True:
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            token_str = model.to_string(next_token)[0]

            if token_str[-1].isspace() or token_str not in non_problematic_tokens:
                break
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

    final_text = model.to_string(output_tokens)[0][13:]
    return final_text, answer_without_number


def generate_full_answer_and_get_logits(model, prompt: str, max_new_tokens: int = 500):
    """
    Generates the full answer using a CoT prompt and returns the logits of the first 
    token immediately following 'The answer is '.
    """
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
            if current_text.endswith("The answer is ") or current_text.endswith("The answer is -"):
                logits = model(output_tokens[:, -2048:])
                answer_token_logits = logits[0, -1, :].clone()
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)
                break

        non_problematic_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", ",", "-"]

        while True:
            logits = model(output_tokens[:, -2048:])
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            token_str = model.to_string(next_token)[0]

            if token_str[-1].isspace() or token_str not in non_problematic_tokens:
                break
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)

    if answer_token_logits is None:
        raise ValueError(
            "Could not find 'The answer is ' in the generated text. "
            "Make sure the prompt leads to a response containing 'The answer is '."
        )

    full_text = model.to_string(output_tokens)[0]
    prompt_text = model.to_string(tokens)[0]
    if full_text.startswith(prompt_text):
        full_answer_text = full_text[len(prompt_text):]
    else:
        full_answer_text = full_text[13:] if len(full_text) > 13 else full_text

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