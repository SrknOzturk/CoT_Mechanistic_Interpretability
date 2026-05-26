"""
src/metrics.py
This module contains metric factories used during activation patching to evaluate
how closely the patched model's output distribution matches a reference distribution.
"""

import torch
import torch.nn.functional as F
from typing import Callable

# ---------------------------------------------------------------------------
# KL Divergence & Entropy Metrics
# ---------------------------------------------------------------------------
def make_kl_metric(clean_logits: torch.Tensor, negate: bool = True):
    """
    Creates a callable metric function that computes the Kullback–Leibler (KL) divergence
    between the clean (reference) model's output distribution and a patched/corrupted model's
    output distribution at the *last* position.

    Parameters
    ----------
    clean_logits : torch.Tensor
        Expected shape: [batch_size, seq_len, vocab_size].
    negate : bool, optional (default=True)
        If True, returns the *negative* KL divergence.

    Returns
    -------
    get_kl_divergence : Callable[[torch.Tensor], torch.Tensor]
        A function that takes the patched logits and returns a scalar KL divergence value 
        between the two distributions: D_KL(P_clean || Q_patched)
    """

    def get_kl_divergence(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute D_KL(P_clean || Q_patched) for the last sequence position.
        """
        # Compute probability distributions
        p = F.softmax(clean_logits[0, -1, :], dim=-1)
        q = F.softmax(patched_logits[0, -1, :], dim=-1)

        # Compute KL divergence (P || Q)
        kl_value = F.kl_div(q.log(), p, reduction='sum')  # Input=log(Q), Target=P

        # Optionally negate so higher value means higher similarity
        if negate:
            kl_value = -kl_value

        return kl_value

    return get_kl_divergence


def make_entropy_difference_metric(clean_logits: torch.Tensor, negate=True):
    """
    Factory function to measure the reduction in entropy after patching.
    Formula: Diff = Clean_Entropy - Patched_Entropy
    """
    # 1. Calculate Baseline (Clean) Entropy
    p = F.softmax(clean_logits[0, -1, :], dim=-1).detach()
    p_log = F.log_softmax(clean_logits[0, -1, :], dim=-1).detach()

    # Shannon Entropy = - sum( p * log(p) )
    clean_entropy = -torch.sum(p * p_log)

    def get_entropy_difference(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the difference: (Clean Entropy) - (Patched Entropy)
        """
        # 2. Calculate Patched Entropy of the current patched run
        q = F.softmax(patched_logits[0, -1, :], dim=-1)
        q_log = F.log_softmax(patched_logits[0, -1, :], dim=-1)

        patched_entropy = -torch.sum(q * q_log)

        # 3. Return the Difference (Value = Clean - Patched)
        result = clean_entropy - patched_entropy
        if negate:
            result = -result
        return result

    return get_entropy_difference


def make_relative_kl_metric(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, negate: bool = True):
    """
    Creates a callable metric that compares how much a patched output improves over a corrupted one
    in terms of KL-divergence to the clean reference.

    Formula:
        metric = ( KL(clean || patched) - KL(clean || corrupted) ) / KL(clean || corrupted)
    """
    # Convert clean and corrupted to probability distributions (last token)
    p = F.softmax(clean_logits[0, -1, :], dim=-1)
    q_corr = F.softmax(corrupted_logits[0, -1, :], dim=-1)

    # KL(clean || corrupted)
    kl_corrupted = F.kl_div(q_corr.log(), p, reduction='sum')

    def get_relative_kl(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute relative KL improvement for patched logits.
        """
        q_patch = F.softmax(patched_logits[0, -1, :], dim=-1)
        kl_patched = F.kl_div(q_patch.log(), p, reduction='sum')

        metric = (kl_patched - kl_corrupted) / (kl_corrupted + 1e-12)  # avoid div by 0

        if negate:
            metric = -metric  # higher = more improvement (less divergence)

        return metric

    return get_relative_kl


def make_relative_entropy_difference_metric(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor):
    """
    Factory function to measure 'Entropy Recovery'.
    It calculates what percentage of the entropy was restored from the
    corrupted state back to the clean state.
    """
    # 1. Calculate Clean Entropy (Baseline Goal)
    p_clean = F.softmax(clean_logits[0, -1, :], dim=-1).detach()
    p_log_clean = F.log_softmax(clean_logits[0, -1, :], dim=-1).detach()
    clean_entropy = -torch.sum(p_clean * p_log_clean)

    # 2. Calculate Corrupted Entropy (Baseline Bad State)
    p_corrupted = F.softmax(corrupted_logits[0, -1, :], dim=-1).detach()
    p_log_corrupted = F.log_softmax(corrupted_logits[0, -1, :], dim=-1).detach()
    corrupted_entropy = -torch.sum(p_corrupted * p_log_corrupted)

    # Denominator represents the "Total Possible Improvement".
    total_entropy_gap = clean_entropy - corrupted_entropy

    if torch.abs(total_entropy_gap) < 1e-5:
        print("Warning: Clean and Corrupted entropies are almost identical. Metric may be unstable.")

    def get_relative_difference(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized score:
        0.0 = No effect (Still behaves like Corrupted)
        1.0 = Full recovery (Behaves like Clean)
        """
        # 3. Calculate Patched Entropy
        q = F.softmax(patched_logits[0, -1, :], dim=-1)
        q_log = F.log_softmax(patched_logits[0, -1, :], dim=-1)
        patched_entropy = -torch.sum(q * q_log)

        # 4. Calculate Normalized Metric: (Patched - Corrupted) / (Clean - Corrupted)
        metric = (patched_entropy - corrupted_entropy) / (total_entropy_gap + 1e-8)

        return metric

    return get_relative_difference


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence (JSD) & Margin Metrics
# ---------------------------------------------------------------------------
def make_jsd_metric(no_cot_logits: torch.Tensor): 
    """
    Stage 1: Called with 'no_cot_logits' inside multi_head_patching.
    We don't strictly need baseline logits for JSD, but we include this 
    wrapper layer to preserve the pipeline architecture and ensure backward compatibility.
    """
    
    def metric_factory(clean_logits: torch.Tensor):
        """
        Stage 2: Called with 'clean_logits' inside the patching_pipeline.
        """
        # Dimensionality check (Ensure it's 1D if a 3D tensor is passed)
        if clean_logits.dim() == 3:
            clean_logits = clean_logits[0, -1, :]
            
        p = F.softmax(clean_logits, dim=-1).detach()
        
        def get_jsd(patched_logits: torch.Tensor) -> torch.Tensor:
            """
            Stage 3: Called with 'patched_logits' inside the iteration loop to produce the score.
            """
            if patched_logits.dim() == 3:
                patched_logits = patched_logits[0, -1, :]
                
            q = F.softmax(patched_logits, dim=-1)
            m = 0.5 * (p + q)
            
            # KL calculations
            kl_p_m = F.kl_div(m.log(), p, reduction='sum')
            kl_q_m = F.kl_div(m.log(), q, reduction='sum')
            
            jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
            return jsd
            
        return get_jsd
        
    return metric_factory


def logit_margin_at_true(logits_1d: torch.Tensor, t_true: int) -> torch.Tensor:
    """
    Calculates the margin between the target token and the next most likely token.
    Margin = logit[t_true] - max_{t != t_true} logit[t].
    logits_1d: Expected shape [vocab_size]
    """
    masked = logits_1d.clone()
    masked[t_true] = float("-inf")
    return logits_1d[t_true] - masked.max()


def margin_recovery_ratio(
    clean_logits_1d: torch.Tensor,
    nocot_logits_1d: torch.Tensor,
    patched_logits_1d: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    t_true = argmax(clean). Margins = logit(t_true) - max_{t!=t_true} logit(t).
    Score = (m_patch - m_nocot) / (m_cot - m_nocot).

    Higher is better: 0 = no improvement over no-COT; 1 ≈ patch matches COT margin.
    """
    t_true = int(clean_logits_1d.argmax(dim=-1).item())
    device = clean_logits_1d.device
    dtype = clean_logits_1d.dtype
    
    m_cot = logit_margin_at_true(clean_logits_1d, t_true)
    m_noc = logit_margin_at_true(nocot_logits_1d.to(device), t_true)
    m_pat = logit_margin_at_true(patched_logits_1d, t_true)
    
    denom = m_cot - m_noc
    if denom.abs() < eps:
        return torch.zeros((), device=device, dtype=dtype)
    return (m_pat - m_noc) / denom


def make_margin_recovery_ratio_metric(nocot_logits_1d: torch.Tensor, eps: float = 1e-8):
    """
    Patching metric factory for Margin Recovery Ratio (MRR).
    patching_pipeline calls: factory(clean_logits_batch).

    clean_logits_batch: [batch, seq, vocab] (reference COT logits expanded).
    Returns get_score(patched_logits) -> scalar (higher = better).
    """
    nocot_stored = nocot_logits_1d

    def factory(clean_logits_batch: torch.Tensor):
        clean_last = clean_logits_batch[0, -1, :].detach()
        t_true = int(clean_last.argmax(dim=-1).item())
        m_cot = logit_margin_at_true(clean_last, t_true)
        m_noc = logit_margin_at_true(nocot_stored.to(clean_last.device), t_true)
        denom = m_cot - m_noc

        def get_score(patched_logits: torch.Tensor) -> torch.Tensor:
            m_pat = logit_margin_at_true(patched_logits[0, -1, :], t_true)
            numer = m_pat - m_noc
            if denom.abs() < eps:
                return torch.zeros((), device=m_pat.device, dtype=m_pat.dtype)
            return numer / denom

        return get_score

    return factory