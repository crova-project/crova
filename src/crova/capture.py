"""Greedy generation and per-position logit capture.

Two execution modes (config key `mode`), which compute the same function with
different floating-point rounding paths:
  teacher_forced  generation with the KV cache; logits from one forward over
                  prompt + response[:-1] (no cache), all positions at once
  prefix          generation and logits both from an independent full forward
                  (no cache) over each prefix, one forward per response token
Use the same mode on the reference and target GPUs.

generate  one greedy response per case (batch size 1), stopping at the first EOS
          or after `max_new_tokens` tokens.
          -> responses/<case>.json  {"case_id", "response", "eos"}
forward   logits at every response position of a given response.
          -> <case>.safetensors  logits [m, vocab], response [m],
             router_logits [layers, m, experts] for MoE models,
             hidden [m, d] (input of the output head) with save_hidden
          With `reference`, only per-case comparison statistics are written
          (<case>.stats.json), so the target logits never touch disk.
          With `topk`, only the top-K log-probabilities of the full-vocabulary
          softmax are written (topk_logprob, topk_index), e.g. for distillation.

Run `generate` and `forward` on the reference GPU first; run `forward` on the
target GPU with `responses` pointing at the reference responses.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import io, metrics
from .models import environment, experts_per_token, load_model
from .workload import Workload


def _model(config):
    return load_model(config["model"], precision=config.get("precision", "bf16"),
                      load=config.get("load", "cast"), selective=config.get("selective"),
                      adapter=config.get("adapter"), device=config.get("device"))


def _cases(config, workload):
    ids = workload.ids(config.get("splits", ["train", "development"]))
    shard, shards = config.get("shard", 0), config.get("num_shards", 1)
    return [cid for i, cid in enumerate(ids) if i % shards == shard]


MODES = ("teacher_forced", "prefix")


def _device(model):
    return next(model.parameters()).device


def mode_of(config):
    mode = config.get("mode", "teacher_forced")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


def greedy(model, prompt, eos, max_new_tokens, mode):
    """Greedy response token IDs, cut after the first EOS."""
    ids = torch.tensor([prompt], device=_device(model))
    if mode == "teacher_forced":
        output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=eos,
                                pad_token_id=eos, use_cache=True)
        response = output[0, ids.shape[1]:].tolist()
        return response[:response.index(eos) + 1] if eos in response else response
    response = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
        token = int(logits.argmax())
        response.append(token)
        if token == eos:
            break
        ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
    return response


@torch.no_grad()
def generate(config):
    workload = Workload(config["workload"])
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    model = _model(config)
    io.write_json(out / f"environment-{config.get('shard', 0)}.json", environment())
    eos, cap, mode = workload.eos_token_id, config.get("max_new_tokens", 1024), mode_of(config)
    started = time.time()
    cases = _cases(config, workload)
    for k, cid in enumerate(cases):
        path = io.case_file(out, cid, ".json")
        if path.exists():
            continue
        response = greedy(model, workload.cases[cid]["input_ids"], eos, cap, mode)
        io.write_json(path, {"case_id": cid, "response": response,
                             "eos": bool(response) and response[-1] == eos})
        print(f"[generate] {k + 1}/{len(cases)} {cid} tokens={len(response)} "
              f"{time.time() - started:.0f}s", flush=True)


def _run(model, ids, keep, *, save_hidden, moe):
    """One forward without cache; returns the last `keep` positions' outputs."""
    head_inputs = []
    handle = None
    if save_hidden:
        head = model.get_output_embeddings()
        handle = head.register_forward_pre_hook(lambda _m, args: head_inputs.append(args[0]))
    try:
        kwargs = {"output_router_logits": True} if moe else {}
        output = model(input_ids=ids, use_cache=False, logits_to_keep=keep, **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    result = {"logits": output.logits[0, -keep:]}
    router = getattr(output, "router_logits", None) if moe else None
    if router is not None:
        result["router_logits"] = torch.stack([r.reshape(-1, r.shape[-1])[-keep:] for r in router])
    if save_hidden:
        result["hidden"] = head_inputs[-1][0, -keep:]
    return result


def position_logits(model, prompt, response, mode="teacher_forced", *, save_hidden=False, moe=False):
    """Logits (and optionally router logits and head inputs) at every response position."""
    m = len(response)
    if mode == "teacher_forced":
        parts = [_run(model, torch.tensor([prompt + response[:-1]], device=_device(model)), m,
                      save_hidden=save_hidden, moe=moe)]
    else:
        parts = [_run(model, torch.tensor([prompt + response[:i]], device=_device(model)), 1,
                      save_hidden=save_hidden, moe=moe) for i in range(m)]
    result = {key: torch.cat([p[key] for p in parts], dim=1 if key == "router_logits" else 0)
                   .contiguous().cpu() for key in parts[0]}
    result["response"] = torch.tensor(response, dtype=torch.int32)
    return result


@torch.no_grad()
def forward(config):
    workload = Workload(config["workload"])
    responses = Path(config["responses"])
    reference = Path(config["reference"]) if config.get("reference") else None
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    model = _model(config)
    moe = experts_per_token(model.config, config["model"]) > 0
    k_experts = experts_per_token(model.config, config["model"])
    io.write_json(out / f"environment-{config.get('shard', 0)}.json", environment())
    mode = mode_of(config)
    started = time.time()
    cases = _cases(config, workload)
    for k, cid in enumerate(cases):
        target = io.case_file(out, cid, ".stats.json" if reference else ".safetensors")
        if target.exists():
            continue
        response = io.read_json(io.case_file(responses, cid, ".json"))["response"]
        values = position_logits(model, workload.cases[cid]["input_ids"], response, mode,
                                 save_hidden=config.get("save_hidden", False), moe=moe)
        if reference:
            ref = load_file(str(io.case_file(reference, cid, ".safetensors")))
            stats = metrics.case_stats(ref, values, experts_per_token=k_experts)
            if config.get("round_to_bf16"):
                rounded = {**values, "logits": values["logits"].to(torch.bfloat16)}
                stats["rounded_bf16"] = metrics.case_stats(ref, rounded,
                                                           experts_per_token=k_experts)
            io.write_json(target, stats)
        else:
            if config.get("topk"):
                logprobs = values.pop("logits").float().log_softmax(-1)
                top, index = logprobs.topk(config["topk"], dim=-1)
                values.update(topk_logprob=top.contiguous(), topk_index=index.int().contiguous())
            save_file(values, str(target))
        if k % 10 == 0:
            print(f"[forward] {k + 1}/{len(cases)} positions={len(response)} "
                  f"{time.time() - started:.0f}s", flush=True)


def compare(config):
    """Pool per-case statistics (or compare two logit directories) into one summary."""
    workload = Workload(config["workload"])
    ids = workload.ids(config.get("splits", ["development"]))
    per_case = {}
    if config.get("stats"):
        for cid in ids:
            stats = io.read_json(io.case_file(config["stats"], cid, ".stats.json"))
            per_case[cid] = stats["rounded_bf16"] if config.get("use_rounded_bf16") else stats
    else:
        k_experts = config.get("experts_per_token", 8)
        for cid in ids:
            per_case[cid] = metrics.case_stats(
                load_file(str(io.case_file(config["reference"], cid, ".safetensors"))),
                load_file(str(io.case_file(config["target"], cid, ".safetensors"))),
                experts_per_token=k_experts)
    rollouts = None
    if config.get("reference_responses") and config.get("target_responses"):
        rollouts = {cid: metrics.rollout_stats(
            io.read_json(io.case_file(config["reference_responses"], cid, ".json"))["response"],
            io.read_json(io.case_file(config["target_responses"], cid, ".json"))["response"])
            for cid in ids}
    summary = metrics.summarize(per_case, rollouts)
    io.write_json(config["output"], {"summary": summary, "per_case": per_case,
                                     "rollouts": rollouts})
    return summary
