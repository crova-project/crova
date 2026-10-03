"""Runtime and peak GPU memory of one model configuration.

Two phases, each with a fresh model load on one GPU:
  benchmark    the complete accuracy benchmark (accuracy.py scoring)
  development  forwards over the reference development responses (in `mode`)
Outputs are computed and discarded.
"""
from __future__ import annotations

import time

import torch

from . import io
from .accuracy import letter_ids, questions, score
from .capture import mode_of, position_logits
from .models import environment, experts_per_token, load_model, load_tokenizer
from .workload import Workload


def _load(config):
    torch.cuda.synchronize()
    started = time.time()
    model = load_model(config["model"], precision=config.get("precision", "bf16"),
                       load=config.get("load", "cast"), selective=config.get("selective"),
                       adapter=config.get("adapter"), device=config.get("device"))
    torch.cuda.synchronize()
    return model, time.time() - started


@torch.no_grad()
def measure(config):
    """config keys: model, precision, load, selective, adapter, benchmarks, prompt,
    workload, responses (reference responses), output."""
    tokenizer = load_tokenizer(config["model"])
    record = {"config": config, "environment": environment()}

    model, record["benchmark_load_seconds"] = _load(config)
    torch.cuda.reset_peak_memory_stats()
    letters = letter_ids(tokenizer)
    items = questions(config, tokenizer)
    started = time.time()
    correct = 0
    for row, input_ids in items:
        predicted, _ = score(model, tokenizer, row, input_ids, config["prompt"], letters=letters,
                             use_cache=mode_of(config) == "teacher_forced")
        correct += predicted == row["answer"]
    torch.cuda.synchronize()
    record.update(benchmark_seconds=time.time() - started, benchmark_questions=len(items),
                  benchmark_correct=correct,
                  benchmark_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  benchmark_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del model
    torch.cuda.empty_cache()

    workload = Workload(config["workload"])
    model, record["development_load_seconds"] = _load(config)
    moe = experts_per_token(model.config, config["model"]) > 0
    torch.cuda.reset_peak_memory_stats()
    started, positions = time.time(), 0
    for cid in workload.ids("development"):
        response = io.read_json(io.case_file(config["responses"], cid, ".json"))["response"]
        position_logits(model, workload.cases[cid]["input_ids"], response, mode_of(config), moe=moe)
        positions += len(response)
    torch.cuda.synchronize()
    record.update(development_seconds=time.time() - started, development_positions=positions,
                  development_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  development_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    record["compute_seconds"] = record["benchmark_seconds"] + record["development_seconds"]
    io.write_json(config["output"], record)
    return record
