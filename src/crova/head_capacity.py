"""How much of the cross-vendor logit mismatch a linear map of the head input can remove.

With H [positions, d] the target GPU's output-head inputs and E [positions, vocab]
the residual (reference logits - target logits) on the same positions, the best
linear correction H @ W removes the part of E in the column space of H. Among maps
of rank r, the best one keeps the r largest singular directions of U^T E, where
U is an orthonormal basis of H's column space. We report

  rank_r_fraction  share of ||E||^2 removable by the best rank-r map
  any_rank_fraction  share removable by an unrestricted linear map

This is a same-data capacity: it measures what a head correction could fit on
these positions, not what it generalises to.
"""
from __future__ import annotations

import torch
from safetensors.torch import load_file

from . import io
from .workload import Workload


def capacity(hidden_blocks, residual_blocks, rank):
    """hidden_blocks / residual_blocks: matching lists of per-case tensors."""
    hidden = torch.cat([h.float() for h in hidden_blocks])
    u, s, _ = torch.linalg.svd(hidden, full_matrices=False)
    tolerance = s.max() * max(hidden.shape) * torch.finfo(torch.float32).eps
    u = u[:, s > tolerance]
    k, energy, start = None, 0.0, 0
    for block in residual_blocks:
        e = block.float()
        rows = u[start:start + len(e)]
        k = rows.T @ e if k is None else k + rows.T @ e
        energy += float(e.square().sum())
        start += len(e)
    singular = torch.linalg.eigvalsh(k @ k.T).flip(0).clamp_min(0)
    return {"positions": start, "retained_directions": u.shape[1],
            "residual_energy": energy, "rank": rank,
            "rank_r_fraction": float(singular[:rank].sum()) / energy,
            "any_rank_fraction": float(singular.sum()) / energy}


def run(config):
    """config keys: workload, target (forward directory with save_hidden), reference
    (reference forward directory), splits (["train"]), rank (32), output."""
    workload = Workload(config["workload"])
    hidden, residual = [], []
    for cid in workload.ids(config.get("splits", ["train"])):
        target = load_file(str(io.case_file(config["target"], cid, ".safetensors")))
        reference = load_file(str(io.case_file(config["reference"], cid, ".safetensors")))
        hidden.append(target["hidden"])
        residual.append(reference["logits"].float() - target["logits"].float())
    result = capacity(hidden, residual, config.get("rank", 32))
    io.write_json(config["output"], result)
    return result
