import torch
import torch.nn.functional as F
from typing import Callable

def make_kl_metric(clean_logits: torch.Tensor, negate: bool = True):
    """
    Creates a callable metric function that computes the Kullback–Leibler (KL) divergence
    between the clean (reference) model's output distribution and a patched/corrupted model's
    output distribution at the *last* position.

    Parameters
    ----------
    clean_logits : torch.Tensor
        Expected shape: [batch_size, seq_len, vocab_size].
    negate : bool, optional (default=False)
        If True, returns the *negative* KL divergence.

    Returns
    -------
    get_kl_divergence : Callable[[torch.Tensor], torch.Tensor]
        A function that takes the patched logits and returns a scalar KL divergence value between the two distributions: D_KL(P_clean || Q_patched)
    """

    def get_kl_divergence(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute D_KL(P_clean || Q_patched) for the last sequence position.
        """
        # Compute probability distributions ---
        p = F.softmax(clean_logits[0, -1, :], dim=-1)
        q = F.softmax(patched_logits[0, -1, :], dim=-1)

        # Compute KL divergence (P || Q)
        kl_value = F.kl_div(q.log(), p, reduction='sum')  

        if negate:
            kl_value = -kl_value

        return kl_value

    return get_kl_divergence

def make_entropy_difference_metric(clean_logits: torch.Tensor, negate = True):
    """
    Factory function to measure the reduction in entropy after patching.
    Formula: Diff = Clean_Entropy - Patched_Entropy
    """

    p = F.softmax(clean_logits[0, -1, :], dim=-1).detach()
    p_log = F.log_softmax(clean_logits[0, -1, :], dim=-1).detach()

    # Shannon Entropy = - sum( p * log(p) )
    clean_entropy = -torch.sum(p * p_log)

    def get_entropy_difference(patched_logits: torch.Tensor) -> torch.Tensor:
        """
        Calculates the difference: (Clean Entropy) - (Patched Entropy)
        """
        # We calculate the entropy of the current patched run.
        q = F.softmax(patched_logits[0, -1, :], dim=-1)
        q_log = F.log_softmax(patched_logits[0, -1, :], dim=-1)

        patched_entropy = -torch.sum(q * q_log)

        # Value = Clean - Patched
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

    Parameters
    ----------
    clean_logits : torch.Tensor
        Reference model logits, shape [batch_size, seq_len, vocab_size].
    corrupted_logits : torch.Tensor
        Corrupted model logits, shape [batch_size, seq_len, vocab_size].
    negate : bool, optional (default=True)
        If True, the metric is negated so that higher values indicate higher similarity to clean logits.

    Returns
    -------
    get_relative_kl : Callable[[torch.Tensor], torch.Tensor]
        Function that takes patched logits and returns the relative KL improvement metric.
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




def normalized_logit_difference(clean_logits, corrupted_logits, answer_token_indices):
    """
    Returns a metric function that calculates normalized Logit Difference recovery.
    answer_token_indices: Tuple/List of (clean_answer_token_id, corrupted_answer_token_id)
    """
    clean_idx, corr_idx = answer_token_indices

    # Helper to calculate Logit Diff
    def get_ld(logits):
        last_token_logits = logits[:, -1, :]
        return last_token_logits[:, clean_idx] - last_token_logits[:, corr_idx]

    # Pre-calculate baselines
    clean_ld = get_ld(clean_logits).item()
    corrupted_ld = get_ld(corrupted_logits).item()

    def metric_fn(patched_logits):
        patched_ld = get_ld(patched_logits).item()
        return (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)

    return metric_fn