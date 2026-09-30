"""Reference-versus-target comparison metrics.

Per case we store sums so that pooled values and question-level bootstrap
intervals can be computed afterwards. All arithmetic is FP32.

  equal_bits        bit-identical BF16 logits (only when both sides are BF16)
  sq_err            sum of squared raw-logit differences
  prob_sq_err       sum of squared probability differences
  kl                sum over positions of KL(p_reference || p_target)
  top1, ord5, set5  argmax, ordered top-5 and top-5 set agreement counts
  router_*          MoE only: agreement of the selected experts per layer and position
"""
from __future__ import annotations

import numpy as np
import torch

BLOCK = 64  # positions per chunk, bounds memory for large vocabularies


def case_stats(reference, target, *, experts_per_token=8):
    zr, zt = reference["logits"], target["logits"]
    if zr.shape != zt.shape:
        raise ValueError(f"logit shapes differ: {tuple(zr.shape)} vs {tuple(zt.shape)}")
    if not torch.equal(reference["response"].long(), target["response"].long()):
        raise ValueError("reference and target were run on different responses")
    m, vocab = zr.shape
    s = {"positions": m, "vocab": vocab, "sq_err": 0.0, "prob_sq_err": 0.0, "kl": 0.0,
         "top1": 0, "ord5": 0, "set5": 0,
         "nonfinite_target": int((~torch.isfinite(zt)).sum()),
         "equal_bits": (int((zr.view(torch.int16) == zt.view(torch.int16)).sum())
                        if zr.dtype == zt.dtype == torch.bfloat16 else None)}
    for start in range(0, m, BLOCK):
        a, b = zr[start:start + BLOCK].float(), zt[start:start + BLOCK].float()
        s["sq_err"] += float((b - a).square().sum())
        lr, lt = a.log_softmax(-1), b.log_softmax(-1)
        pr, pt = lr.exp(), lt.exp()
        s["prob_sq_err"] += float((pt - pr).square().sum())
        s["kl"] += float((pr * (lr - lt)).sum())
        s["top1"] += int((a.argmax(-1) == b.argmax(-1)).sum())
        ir, it = a.topk(5, dim=-1).indices, b.topk(5, dim=-1).indices
        s["ord5"] += int((ir == it).all(-1).sum())
        s["set5"] += int((ir.sort(-1).values == it.sort(-1).values).all(-1).sum())
    if "router_logits" in reference and "router_logits" in target:
        rr, rt = reference["router_logits"].float(), target["router_logits"].float()
        er = rr.topk(experts_per_token, dim=-1).indices  # layers x positions x k
        et = rt.topk(experts_per_token, dim=-1).indices
        same_set = (er.sort(-1).values == et.sort(-1).values).all(-1)
        same_order = (er == et).all(-1)
        s.update(router_slots=int(same_set.numel()), router_set_equal=int(same_set.sum()),
                 router_ord_equal=int(same_order.sum()),
                 router_positions_all_layers_equal=int(same_set.all(0).sum()),
                 router_logit_sq_err=float((rr - rt).square().sum()),
                 router_logit_count=int(rr.numel()))
    return s


def rollout_stats(reference_tokens, target_tokens):
    """Whole-response agreement between two greedy generations."""
    first = next((i for i, (x, y) in enumerate(zip(reference_tokens, target_tokens)) if x != y),
                 min(len(reference_tokens), len(target_tokens)))
    return {"equal": reference_tokens == target_tokens, "first_divergence": first,
            "reference_length": len(reference_tokens), "target_length": len(target_tokens)}


def _values(per_case, names, key):
    return np.array([per_case[n][key] for n in names], dtype=np.float64)


def summarize(per_case, rollouts=None, *, resamples=10000, seed=20260820):
    """Pooled metrics with 95% whole-question bootstrap intervals."""
    names = sorted(per_case)
    positions = _values(per_case, names, "positions")
    vocab = per_case[names[0]]["vocab"]
    result = {"cases": len(names), "positions": int(positions.sum())}
    index = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    for key, (numerator, denominator) in {"raw_logit_mse": ("sq_err", positions * vocab),
                                          "prob_mse": ("prob_sq_err", positions * vocab),
                                          "forward_kl": ("kl", positions)}.items():
        x = _values(per_case, names, numerator)
        result[key] = float(x.sum() / denominator.sum())
        draws = x[index].sum(1) / denominator[index].sum(1)
        result[key + "_ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    for key in ("top1", "ord5", "set5"):
        total = _values(per_case, names, key).sum()
        result[key] = int(total)
        result[key + "_pct"] = float(100 * total / positions.sum())
    if per_case[names[0]]["equal_bits"] is not None:
        total = _values(per_case, names, "equal_bits").sum()
        result["equal_bits"] = int(total)
        result["equal_bits_pct"] = float(100 * total / (positions.sum() * vocab))
    result["nonfinite_target"] = int(_values(per_case, names, "nonfinite_target").sum())
    if "router_slots" in per_case[names[0]]:
        slots = _values(per_case, names, "router_slots").sum()
        for key in ("router_set_equal", "router_ord_equal"):
            result[key + "_pct"] = float(100 * _values(per_case, names, key).sum() / slots)
        result["router_positions_all_layers_equal_pct"] = float(
            100 * _values(per_case, names, "router_positions_all_layers_equal").sum() / positions.sum())
        result["router_logit_mse"] = float(_values(per_case, names, "router_logit_sq_err").sum()
                                           / _values(per_case, names, "router_logit_count").sum())
    if rollouts:
        equal = [r["equal"] for r in rollouts.values()]
        result.update(rollout_equal=int(sum(equal)), rollout_cases=len(equal),
                      rollout_equal_pct=100 * sum(equal) / len(equal),
                      rollout_median_first_divergence=float(
                          np.median([r["first_divergence"] for r in rollouts.values()])))
    return result


def paired_bootstrap(a, b, *, resamples=10000, seed=20260820):
    """Target B versus target A, both measured against the same reference.

    Returns [point, 2.5%, 97.5%] for the relative change in each pooled error
    (percent) and the difference in each agreement count.
    """
    names = sorted(set(a) & set(b))
    if len(names) != len(a) or len(names) != len(b):
        raise ValueError("paired bootstrap needs the same cases on both sides")
    index = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    out = {}
    for key in ("sq_err", "kl", "prob_sq_err"):
        x, y = _values(a, names, key), _values(b, names, key)
        draws = 100 * (y[index].sum(1) / x[index].sum(1) - 1)
        out[key + "_rel_change_pct"] = [float(100 * (y.sum() / x.sum() - 1)),
                                        float(np.percentile(draws, 2.5)),
                                        float(np.percentile(draws, 97.5))]
    for key in ("top1", "ord5", "set5", "equal_bits", "router_set_equal"):
        if a[names[0]].get(key) is None or b[names[0]].get(key) is None:
            continue
        d = _values(b, names, key) - _values(a, names, key)
        draws = d[index].sum(1)
        out[key + "_diff"] = [int(d.sum()), float(np.percentile(draws, 2.5)),
                              float(np.percentile(draws, 97.5))]
    return out
