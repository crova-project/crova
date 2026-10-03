"""Downstream knowledge distillation with teachers run on different GPUs.

Only the teacher distribution differs between arms: every student starts from
the same checkpoint and sees the same reference-GPU responses, optimizer,
schedule and number of updates.

topk       reduce teacher logits (capture.forward output) to the top-K
           log-probabilities of the full-vocabulary softmax
train      loss at each response position: KL(q || p_student) over the teacher
           top-K tokens, with q the renormalised teacher top-K probabilities
evaluate   compare students with a reference student on the development responses
agreement  fraction of benchmark questions where two students predict the same answer
compare_targets  overlap and total variation between two teachers' top-K targets
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from . import io
from .models import environment, load_model, numerics, resolve
from .workload import Workload


def topk(config):
    """config keys: teacher (forward directory), workload, output, k (64), splits."""
    workload = Workload(config["workload"])
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    for cid in workload.ids(config.get("splits", ["train"])):
        target = io.case_file(out, cid, ".safetensors")
        if target.exists():
            continue
        values = load_file(str(io.case_file(config["teacher"], cid, ".safetensors")))
        logprobs = values["logits"].float().log_softmax(-1)
        top, index = logprobs.topk(config.get("k", 64), dim=-1)
        save_file({"topk_logprob": top.contiguous(), "topk_index": index.int().contiguous(),
                   "response": values["response"]}, str(target))


def train(config):
    """config keys: student, teacher (top-K directory), workload, output, seed (data
    order), epochs (2), lr (1e-5), accumulation (8), warmup (8), wandb (optional)."""
    numerics()
    torch.manual_seed(0)
    workload = Workload(config["workload"])
    data = []
    for cid in workload.ids("train"):
        t = load_file(str(io.case_file(config["teacher"], cid, ".safetensors")))
        logprob = t["topk_logprob"].float()
        if not torch.isfinite(logprob).all():
            raise ValueError(f"teacher distribution for {cid} is not finite; this teacher is unusable")
        data.append((workload.cases[cid]["input_ids"], t["response"].tolist(),
                     t["topk_index"].long(), logprob.softmax(-1)))
    from transformers import AutoModelForCausalLM

    model_id, revision, _ = resolve(config["student"])
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32,
                                                 attn_implementation="eager").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    epochs, accumulation = config.get("epochs", 2), config.get("accumulation", 8)
    warmup = config.get("warmup", 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-5), betas=(0.9, 0.95),
                                  weight_decay=0.0)
    steps = epochs * len(data) // accumulation
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup)
                                                  * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    rng, started, step = random.Random(config.get("seed", 0)), time.time(), 0
    model.train()
    with (out / "progress.jsonl").open("w") as log:
        for epoch in range(epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            for j, i in enumerate(order):
                prompt, response, index, q = data[i]
                ids = torch.tensor([prompt + response[:-1]], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids).logits[0, -len(response):]
                student = logits.float().log_softmax(-1).gather(-1, index.cuda())
                q = q.cuda()
                loss = (q * (q.clamp_min(1e-30).log() - student)).sum(-1).mean()
                (loss / accumulation).backward()
                if (j + 1) % accumulation == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item(),
                                          "grad_norm": norm.item(),
                                          "lr": scheduler.get_last_lr()[0],
                                          "seconds": time.time() - started}) + "\n")
                    log.flush()
    model.to(torch.bfloat16).save_pretrained(out / "student", safe_serialization=True)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(out / "student")
    io.write_json(out / "run.json", {"config": config, "updates": step, "environment": environment()})


@torch.no_grad()
def evaluate(config):
    """config keys: students (mapping name -> path; the first is the reference),
    workload, responses (reference-GPU responses), output."""
    workload = Workload(config["workload"])
    names = list(config["students"])
    models = {name: load_model(path, device=config.get("device"))
              for name, path in config["students"].items()}
    per_case = {name: {} for name in names[1:]}
    for cid in workload.ids("development"):
        response = io.read_json(io.case_file(config["responses"], cid, ".json"))["response"]
        device = next(models[names[0]].parameters()).device
        ids = torch.tensor([workload.cases[cid]["input_ids"] + response[:-1]], device=device)
        m = len(response)
        reference = models[names[0]](input_ids=ids, use_cache=False).logits[0, -m:].float()
        log_r = reference.log_softmax(-1)
        for name in names[1:]:
            z = models[name](input_ids=ids, use_cache=False).logits[0, -m:].float()
            per_case[name][cid] = {
                "positions": m, "vocab": z.shape[-1],
                "kl": float((log_r.exp() * (log_r - z.log_softmax(-1))).sum()),
                "top1": int((z.argmax(-1) == reference.argmax(-1)).sum()),
                "sq_err": float((z - reference).square().sum())}
    index = None
    summary = {}
    for name, cases in per_case.items():
        ids = sorted(cases)
        positions = np.array([cases[c]["positions"] for c in ids], float)
        kl = np.array([cases[c]["kl"] for c in ids])
        top1 = np.array([cases[c]["top1"] for c in ids], float)
        if index is None:
            index = np.random.default_rng(20260820).integers(0, len(ids), size=(10000, len(ids)))
        kl_draws = kl[index].sum(1) / positions[index].sum(1)
        top1_draws = top1[index].sum(1) / positions[index].sum(1)
        summary[name] = {
            "cases": len(ids), "positions": int(positions.sum()),
            "kl_to_reference": float(kl.sum() / positions.sum()),
            "kl_ci95": [float(np.percentile(kl_draws, 2.5)), float(np.percentile(kl_draws, 97.5))],
            "top1_agreement_pct": float(100 * top1.sum() / positions.sum()),
            "top1_ci95": [float(100 * np.percentile(top1_draws, 2.5)),
                          float(100 * np.percentile(top1_draws, 97.5))],
            "raw_logit_mse_to_reference": float(sum(cases[c]["sq_err"] for c in ids)
                                                / (positions.sum() * cases[ids[0]]["vocab"]))}
    io.write_json(config["output"], {"reference": names[0], "summary": summary,
                                     "per_case": per_case})
    return summary


def agreement(reference_accuracy, other_accuracy):
    """Per-benchmark share of questions where two accuracy runs predict the same letter."""
    reference = {r["case_id"]: r for r in io.read_json(reference_accuracy)["rows"]}
    other = {r["case_id"]: r for r in io.read_json(other_accuracy)["rows"]}
    result = {}
    for category in sorted({r["category"] for r in reference.values()}):
        ids = [c for c, r in reference.items() if r["category"] == category]
        same = sum(reference[c]["predicted"] == other[c]["predicted"] for c in ids)
        result[category] = {"questions": len(ids), "same": same, "same_pct": 100 * same / len(ids)}
    return result


def compare_targets(reference_dir, other_dir, workload_dir, splits=("train",)):
    """How far another teacher's renormalised top-K target is from the reference teacher's."""
    workload = Workload(workload_dir)
    positions = overlap = same_set = same_order = 0
    total_variation = 0.0
    for cid in workload.ids(list(splits)):
        a = load_file(str(io.case_file(reference_dir, cid, ".safetensors")))
        b = load_file(str(io.case_file(other_dir, cid, ".safetensors")))
        ia, ib = a["topk_index"].long(), b["topk_index"].long()
        pa, pb = a["topk_logprob"].float().softmax(-1), b["topk_logprob"].float().softmax(-1)
        equal = ia.unsqueeze(2) == ib.unsqueeze(1)
        in_b = equal.any(1)
        pb_at_a = (equal.float() * pb.unsqueeze(1)).sum(2)
        tv = 0.5 * ((pa - pb_at_a).abs().sum(-1) + (pb * (~in_b)).sum(-1))
        shared = equal.any(2).sum(-1)
        positions += ia.shape[0]
        overlap += int(shared.sum())
        total_variation += float(tv.sum())
        same_set += int((shared == ia.shape[1]).sum())
        same_order += int((ia == ib).all(-1).sum())
    return {"positions": positions, "mean_overlap": overlap / positions,
            "same_set_pct": 100 * same_set / positions,
            "same_order_pct": 100 * same_order / positions,
            "mean_total_variation": total_variation / positions}
