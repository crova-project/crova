"""The four fidelity losses used to train the output-head LoRA.

All take candidate and reference logits [positions, vocab], compute in FP32 and
return the mean over positions. The reference is never differentiated. The
reference top-5 order breaks exact score ties by ascending token ID.

  raw_logit_mse  mean squared logit difference
  forward_kl     KL(p_reference || p_candidate)
  token_gap      reference-gap loss with k=1
  top5_gap       reference-gap loss with k=5

The reference-gap loss uses the residual d = candidate - reference. For the
reference top-k tokens q_1..q_k it penalises changes in the gaps between
consecutive selected tokens, and between q_k and every token outside the top-k:

  [ sum_i (d[q_i] - d[q_{i+1}])^2 + mean_{j not in top-k} (d[q_k] - d[j])^2 ] / k

It is zero when the candidate equals the reference.
"""
from __future__ import annotations

import torch

NAMES = ("raw_logit_mse", "forward_kl", "token_gap", "top5_gap")

# Loss-weighting profiles: weights for (raw_logit_mse, forward_kl, token_gap, top5_gap).
PROFILES = {
    "equal": (0.25, 0.25, 0.25, 0.25),
    "numerical": (0.4, 0.4, 0.1, 0.1),
    "mse": (0.7, 0.1, 0.1, 0.1),
    "kl": (0.1, 0.7, 0.1, 0.1),
    "token": (0.1, 0.1, 0.7, 0.1),
    "top5": (0.1, 0.1, 0.1, 0.7),
    "decision": (0.1, 0.1, 0.4, 0.4),
    "kl_only": (0.0, 1.0, 0.0, 0.0),
}


def reference_order(reference, k=5):
    return reference.float().argsort(dim=-1, descending=True, stable=True)[..., :k]


def gap_loss(candidate, reference, order):
    """Reference-gap loss for the given reference top-k order [positions, k]."""
    k = order.shape[-1]
    residual = candidate.float() - reference.detach().float()
    selected = residual.gather(-1, order)
    adjacent = (selected[..., :-1] - selected[..., 1:]).square().sum(-1)
    outside = torch.ones_like(residual, dtype=torch.bool).scatter(-1, order, False)
    boundary = (selected[..., -1:] - residual).masked_fill(~outside, 0).square().sum(-1)
    boundary = boundary / (residual.shape[-1] - k)
    return ((adjacent + boundary) / k).mean()


def losses(candidate, reference):
    """Dictionary of the four losses (FP32 scalars)."""
    c, r = candidate.float(), reference.detach().float()
    order = reference_order(r)
    log_r = r.log_softmax(-1)
    return {
        "raw_logit_mse": (c - r).square().mean(),
        "forward_kl": (log_r.exp() * (log_r - c.log_softmax(-1))).sum(-1).mean(),
        "token_gap": gap_loss(c, r, order[..., :1]),
        "top5_gap": gap_loss(c, r, order),
    }


def objective(values, weights, scales):
    """Weighted sum of normalised losses: sum_j w_j * L_j / c_j."""
    return sum(weights[i] * values[name] / scales[name] for i, name in enumerate(NAMES))
