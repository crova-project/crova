"""Multiple-choice accuracy on complete benchmark test sets.

Two scoring methods, chosen by the prompt style:
  plain  prompt ends in "Answer:"; the prediction is the answer letter whose
         token (" A", " B", ...) has the highest next-token logit.
  chat   chat-template prompt asking for one letter; the model generates up to
         8 tokens greedily and the first standalone letter A-E is parsed.
         Unparseable outputs count as wrong and are reported separately.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import torch

from . import io
from .models import environment, load_model, load_tokenizer
from .workload import LETTERS, Workload, load_benchmark, render

CHOICE = re.compile(r"(?<![A-Z0-9])[A-E](?![A-Z0-9])")


def letter_ids(tokenizer):
    return [tokenizer.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]


def score(model, tokenizer, row, input_ids, style, *, letters=None, max_new_tokens=8):
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    if style == "plain":
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        candidates = logits[letters[:len(row["choices"])]]
        return LETTERS[int(candidates.argmax())], bool(torch.isfinite(candidates).all())
    output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    match = CHOICE.search(text)
    return (match.group() if match else None), True


def questions(config, tokenizer):
    rows = [row for category in config["benchmarks"] for row in load_benchmark(category)]
    shard, shards = config.get("shard", 0), config.get("num_shards", 1)
    return [(row, render(tokenizer, row, config["prompt"]))
            for i, row in enumerate(rows) if i % shards == shard]


@torch.no_grad()
def evaluate(config):
    """config keys: model, precision, load, selective, adapter, benchmarks, prompt,
    output, exclude_workload (optional), shard/num_shards (optional)."""
    tokenizer = load_tokenizer(config["model"])
    model = load_model(config["model"], precision=config.get("precision", "bf16"),
                       load=config.get("load", "cast"), selective=config.get("selective"),
                       adapter=config.get("adapter"))
    letters = letter_ids(tokenizer)
    rows, started = [], time.time()
    items = questions(config, tokenizer)
    for k, (row, input_ids) in enumerate(items):
        predicted, finite = score(model, tokenizer, row, input_ids, config["prompt"],
                                  letters=letters)
        rows.append({"case_id": row["case_id"], "category": row["category"],
                     "predicted": predicted, "answer": row["answer"],
                     "correct": predicted == row["answer"], "parsed": predicted is not None,
                     "finite": finite})
        if k % 500 == 0:
            print(f"[accuracy] {k}/{len(items)} {time.time() - started:.0f}s", flush=True)
    result = {"summary": summarize(rows, config.get("exclude_workload")),
              "rows": rows, "environment": environment()}
    io.write_json(config["output"], result)
    return result["summary"]


def summarize(rows, exclude_workload=None):
    excluded = set()
    if exclude_workload:
        excluded = set(Workload(exclude_workload).cases)
    summary = {}
    for category in sorted({r["category"] for r in rows}):
        groups = {"all": [r for r in rows if r["category"] == category]}
        if excluded:
            groups["excluding_workload"] = [r for r in groups["all"] if r["case_id"] not in excluded]
        summary[category] = {
            name: {"cases": len(g), "correct": sum(r["correct"] for r in g),
                   "accuracy": sum(r["correct"] for r in g) / len(g) if g else None,
                   "unparsed": sum(not r["parsed"] for r in g),
                   "nonfinite": sum(not r["finite"] for r in g)}
            for name, g in groups.items()}
    return summary


def merge(paths, output, exclude_workload=None):
    """Combine sharded accuracy outputs into one summary."""
    rows = [row for path in paths for row in io.read_json(path)["rows"]]
    summary = summarize(rows, exclude_workload)
    io.write_json(Path(output), {"summary": summary, "rows": rows})
    return summary
