"""Output-head LoRA trained to match reference-GPU logits on the target GPU.

The base model stays frozen in native BF16. A rank-32 LoRA (alpha 64, FP32
factors, A Kaiming-uniform, B zero) is attached to the output head, so the
corrected logits come out of the same BF16 head as in deployment. The four
losses (losses.py) are computed in FP32, each divided by its mean over all
training positions before training (c_j), and combined with a weighting
profile. Each attempt uses one training response. Its loss is the mean over its
positions times (its length / the mean training response length), so every
training token carries the same weight regardless of response length.

Optimizer: AdamW (lr 1e-4, betas 0.9/0.999, eps 1e-8, no weight decay). Each
proposed update is tried at fractions 1, 1/2, ..., 1/64 of its size and kept at
the first fraction that lowers this response's objective by more than
1e-8 * max(1, |L|), or leaves it unchanged while the update points downhill.
Otherwise the factors and optimizer state are restored.

Because the base model is frozen, the output-head inputs are computed once per
training position (with the same `mode` as the reference logits) and cached.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from . import io
from .capture import mode_of, position_logits
from .losses import NAMES, PROFILES, losses, objective
from .models import environment, load_model
from .workload import Workload

FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
MIN_DECREASE = 1e-8
CHUNK = 128  # positions per loss chunk


def attach(model, *, rank=32, alpha=64, seed=0):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                             bias="none", target_modules=["lm_head"]))
    head = model.get_base_model().lm_head
    a, b = head.lora_A["default"].weight, head.lora_B["default"].weight
    generator = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.empty(a.shape, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(init, a=math.sqrt(5), generator=generator)
    with torch.no_grad():
        a.copy_(init)
        b.zero_()
    for p in model.parameters():
        p.requires_grad_(False)
    params = [a, b]
    for p in params:
        if p.dtype != torch.float32:
            p.data = p.data.float()
        p.requires_grad_(True)
    return model, params


def schedule(cases, epochs, seed=20260910):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [i for _ in range(epochs) for i in torch.randperm(cases, generator=generator).tolist()]


class Data:
    """Cached head inputs and on-disk reference logits for the training responses."""

    def __init__(self, model, workload, reference, max_positions, mode):
        self.reference, self.max_positions = Path(reference), max_positions
        self.names = workload.ids("train")
        self.hidden = []
        with torch.no_grad():
            for k, cid in enumerate(self.names):
                response = load_file(str(io.case_file(reference, cid, ".safetensors")))["response"]
                response = response.tolist()[:max_positions] if max_positions else response.tolist()
                self.hidden.append(position_logits(model, workload.cases[cid]["input_ids"],
                                                   response, mode, save_hidden=True)["hidden"])
                if k % 100 == 0:
                    print(f"[lora] cached head inputs {k + 1}/{len(self.names)}", flush=True)

    def target(self, i):
        logits = load_file(str(io.case_file(self.reference, self.names[i], ".safetensors")))["logits"]
        return logits[:len(self.hidden[i])]


def _loss_terms(head, hidden, target, device, *, weights=None, scales=None, multiplier=1.0,
                backward=False):
    """Mean over positions of each loss, and the weighted objective times `multiplier`."""
    m = len(hidden)
    totals = dict.fromkeys(NAMES, 0.0)
    value = 0.0
    for start in range(0, m, CHUNK):
        h = hidden[start:start + CHUNK].to(device)
        r = target[start:start + CHUNK].to(device)
        share = len(h) / m
        with torch.set_grad_enabled(backward):
            terms = losses(head(h), r)
            if weights is not None:
                loss = objective(terms, weights, scales) * share * multiplier
                if backward:
                    loss.backward()
                value += float(loss.detach())
        for name in NAMES:
            totals[name] += float(terms[name].detach()) * share
    return value, totals


def backtracking_step(params, optimizer, evaluate):
    """One AdamW proposal with step-size backtracking; returns a log record."""
    before = [p.detach().clone() for p in params]
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    value, terms = evaluate(True)
    if not math.isfinite(value):
        raise ValueError("non-finite training loss")
    gradients = [p.grad.detach().clone() for p in params]
    optimizer.step()
    delta = [p.detach() - old for p, old in zip(params, before)]
    accepted, new_value = 0.0, value
    with torch.no_grad():
        for fraction in FRACTIONS:
            for p, old, change in zip(params, before, delta):
                p.copy_(old + fraction * change)
            trial, _ = evaluate(False)
            descent = sum(float((g * (fraction * change)).sum()) for g, change in zip(gradients, delta))
            if trial < value - MIN_DECREASE * max(1.0, abs(value)) or (trial <= value and descent < 0):
                accepted, new_value = fraction, trial
                break
        if not accepted:
            for p, old in zip(params, before):
                p.copy_(old)
            optimizer.load_state_dict(state)
    optimizer.zero_grad(set_to_none=True)
    return {"objective": value, "new_objective": new_value, "fraction": accepted,
            "losses": terms,
            "gradient_norm": math.sqrt(sum(float(g.square().sum()) for g in gradients))}


def train(config):
    """config keys: model, workload, reference (reference-GPU forward directory),
    output, profile, mode (as used for the reference logits), epochs (2),
    max_positions (null = whole response),
    rank (32), alpha (64), lr (1e-4), save_steps (optional list), wandb (optional)."""
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    workload = Workload(config["workload"])
    weights = PROFILES[config["profile"]]
    model = load_model(config["model"])
    model, params = attach(model, rank=config.get("rank", 32), alpha=config.get("alpha", 64))
    device = params[0].device
    head = model.get_base_model().lm_head
    data = Data(model, workload, config["reference"], config.get("max_positions"), mode_of(config))

    # Normalizers: each loss's mean over all training positions before training.
    sums, positions = dict.fromkeys(NAMES, 0.0), 0
    for i in range(len(data.names)):
        _, terms = _loss_terms(head, data.hidden[i], data.target(i), device)
        m = len(data.hidden[i])
        for name in NAMES:
            sums[name] += terms[name] * m
        positions += m
    scales = {name: sums[name] / positions for name in NAMES}
    mean_length = positions / len(data.names)
    io.write_json(out / "normalizers.json", {"scales": scales, "positions": positions,
                                             "cases": len(data.names), "mean_length": mean_length})

    optimizer = torch.optim.AdamW(params, lr=config.get("lr", 1e-4), betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0)
    order = schedule(len(data.names), config.get("epochs", 2))
    save_steps = set(config.get("save_steps", []))
    logger = _wandb(config)
    started = time.time()
    with (out / "progress.jsonl").open("w") as log:
        for step, i in enumerate(order, 1):
            hidden, target = data.hidden[i], data.target(i)

            def evaluate(backward, hidden=hidden, target=target):
                return _loss_terms(head, hidden, target, device, weights=weights, scales=scales,
                                   multiplier=len(hidden) / mean_length, backward=backward)

            record = backtracking_step(params, optimizer, evaluate)
            record.update(step=step, case_id=data.names[i], positions=len(data.hidden[i]),
                          seconds=time.time() - started)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if logger:
                logger.log({"objective": record["objective"], "fraction": record["fraction"],
                            **{f"loss/{k}": v for k, v in record["losses"].items()}}, step=step)
            if step in save_steps:
                model.save_pretrained(out / f"step-{step:04d}", save_embedding_layers=False)
            if step % 100 == 0:
                print(f"[lora] step {step}/{len(order)} objective {record['objective']:.5f} "
                      f"{time.time() - started:.0f}s", flush=True)
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    io.write_json(out / "run.json", {"config": config, "steps": len(order),
                                     "environment": environment()})
    if logger:
        logger.finish()
    return out


def _wandb(config):
    """Optional scalar logging; project and entity come from WANDB_PROJECT/WANDB_ENTITY."""
    if not config.get("wandb"):
        return None
    import wandb

    return wandb.init(config={k: v for k, v in config.items() if k != "wandb"},
                      name=Path(config["output"]).name)
