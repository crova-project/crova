"""Multiple-choice benchmarks, prompt rendering and train/development workloads.

Benchmarks load from the Hugging Face Hub at pinned revisions and are normalised
to rows {case_id, category, subject, question, choices, answer}, where answer is
a letter. Rows keep the dataset order; the few exact duplicates in the test
sets are kept (with a numbered case_id), so counts match the standard test sets.

A workload is a directory with
  manifest.json  model, prompt style, selection settings and
                 splits {"train": [case_id, ...], "development": [...]}
  cases.jsonl    one normalised row per selected case, plus "input_ids"
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import io
from .models import load_tokenizer

LETTERS = "ABCDE"
BENCHMARKS = {  # name: (dataset, config, revision)
    "mmlu": ("cais/mmlu", "all", "c30699e8356da336a370243923dbaf21066bb9fe"),
    "arabicmmlu": ("MBZUAI/ArabicMMLU", "All", "7aa530e2893ac420352b3f5c1a1310c010e9758b"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453"),
}
CHAT_INSTRUCTION = "Answer with one uppercase choice letter only."


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def question_key(row):
    """Identity of the question text shown to the model (question and choices)."""
    return _digest([row["category"], row["question"], row["choices"]])[:16]


def _normalize(category, row):
    if category == "mmlu":
        choices, answer = list(row["choices"]), LETTERS[row["answer"]]
        subject, question = row["subject"], row["question"]
    elif category == "arabicmmlu":
        choices = ["" if row.get(f"Option {i}") is None else str(row[f"Option {i}"]).strip()
                   for i in range(1, 6)]
        while choices and not choices[-1]:
            choices.pop()
        subject, question, answer = row["Subject"], row["Question"], row["Answer Key"]
    elif category == "arc":
        labels, choices = list(row["choices"]["label"]), list(row["choices"]["text"])
        subject, question = "arc-challenge", row["question"]
        answer = LETTERS[labels.index(row["answerKey"])]
    else:
        raise ValueError(f"unknown benchmark: {category}")
    if not 2 <= len(choices) <= 5 or answer not in LETTERS[:len(choices)] or not all(choices):
        raise ValueError(f"malformed {category} row: {question[:60]!r}")
    body = {"category": category, "subject": subject, "question": question, "choices": choices,
            "answer": answer}
    return {"case_id": f"{category}-{_digest(body)[:16]}", **body}


def load_benchmark(category, splits=("test",)):
    from datasets import load_dataset

    dataset_id, config, revision = BENCHMARKS[category]
    rows, seen = [], {}
    source = [row for split in splits
              for row in load_dataset(dataset_id, config, split=split, revision=revision)]
    for row in source:
        row = _normalize(category, row)
        seen[row["case_id"]] = seen.get(row["case_id"], 0) + 1
        if seen[row["case_id"]] > 1:
            row["case_id"] += f"-{seen[row['case_id']]}"
        rows.append(row)
    return rows


def render(tokenizer, row, style):
    """Token IDs for one multiple-choice question.

    chat:  chat template, choices on separate lines and an instruction to answer
           with one letter; scored by generating and parsing a letter.
    plain: "question\\nA. ...\\nB. ...\\nAnswer:"; scored by the letter logits.
    """
    if style == "chat":
        choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        messages = [{"role": "user", "content": f"{row['question']}\n{choices}\n{CHAT_INSTRUCTION}"}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(encoded, dict) or hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        return list(encoded)
    if style == "plain":
        text = (row["question"] + "\n"
                + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(row["choices"])) + "Answer:")
        return tokenizer(text).input_ids
    raise ValueError(f"unknown prompt style: {style}")


def _round_robin(rows, count, seed, tag):
    """Deterministic subject-balanced selection."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append((_digest([seed, tag, row["case_id"]]), row))
    ordered = [[r for _, r in sorted(grouped[s], key=lambda x: x[0])] for s in sorted(grouped)]
    result, index = [], 0
    while len(result) < count:
        progressed = False
        for subject_rows in ordered:
            if index < len(subject_rows):
                result.append(subject_rows[index])
                progressed = True
                if len(result) == count:
                    return result
        if not progressed:
            raise ValueError(f"not enough questions for {tag}: need {count}")
        index += 1
    return result


def build(config):
    """Select train/development questions per benchmark and tokenize them.

    config keys: model, output, benchmarks (list), prompt (chat|plain),
    train_per_benchmark, development_per_benchmark, seed,
    source_splits (optional mapping benchmark -> dataset splits to draw from;
    default test), exclude (optional file with one case_id per line).
    """
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(config["model"])
    seed = str(config.get("seed", 20260910))
    excluded = set()
    if config.get("exclude"):
        excluded = {line.strip() for line in Path(config["exclude"]).read_text().splitlines()
                    if line.strip()}
    splits, cases = {"train": [], "development": []}, []
    for category in config["benchmarks"]:
        splits_for = config.get("source_splits", {}).get(category, ["test"])
        rows = [r for r in load_benchmark(category, splits_for) if r["case_id"] not in excluded]
        dev = _round_robin(rows, config["development_per_benchmark"], seed, "development")
        dev_ids = {r["case_id"] for r in dev}
        train = _round_robin([r for r in rows if r["case_id"] not in dev_ids],
                             config["train_per_benchmark"], seed, "train")
        for split, selected in (("development", dev), ("train", train)):
            for row in selected:
                splits[split].append(row["case_id"])
                cases.append({**row, "input_ids": render(tokenizer, row, config["prompt"])})
    io.write_jsonl(output / "cases.jsonl", cases)
    io.write_json(output / "manifest.json", {
        "model": config["model"], "prompt": config["prompt"], "benchmarks": config["benchmarks"],
        "source_splits": config.get("source_splits", {}),
        "seed": seed, "excluded": len(excluded), "eos_token_id": tokenizer.eos_token_id,
        "splits": splits})
    return output


class Workload:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = io.read_json(self.root / "manifest.json")
        self.cases = {row["case_id"]: row for row in io.read_jsonl(self.root / "cases.jsonl")}

    def ids(self, splits):
        if isinstance(splits, str):
            splits = [splits]
        return [cid for split in splits for cid in self.manifest["splits"][split]]

    @property
    def eos_token_id(self):
        return self.manifest["eos_token_id"]


def _text_key(row):
    return _digest([row["question"], row["choices"]])[:16]


def extend(config):
    """Grow an existing workload's training split, keeping all of its questions.

    config keys: base (workload directory), model, output, train_total,
    sources (mapping benchmark -> dataset splits to draw new questions from),
    exclude_test (benchmarks whose test sets must not overlap), seed.
    New questions are taken in a deterministic hash order, skipping duplicates of
    existing questions and anything whose text matches an excluded test question.
    Every case is tokenized for `model` with the base workload's prompt style.
    """
    base = Workload(config["base"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    seed = str(config.get("seed", 20260910))
    taken = {_text_key(row) for row in base.cases.values()}
    test = {_text_key(row) for category in config.get("exclude_test", [])
            for row in load_benchmark(category)}
    candidates = []
    for category, splits in config["sources"].items():
        for row in load_benchmark(category, splits):
            key = _text_key(row)
            if key not in taken and key not in test:
                taken.add(key)
                candidates.append(row)
    candidates.sort(key=lambda row: _digest([seed, "extend", row["case_id"]]))
    needed = config["train_total"] - len(base.ids("train"))
    if needed < 0 or needed > len(candidates):
        raise ValueError(f"cannot reach train_total={config['train_total']} "
                         f"({len(candidates)} new candidates)")
    new = candidates[:needed]
    tokenizer = load_tokenizer(config["model"])
    prompt = base.manifest["prompt"]
    cases = [{**{k: v for k, v in row.items() if k != "input_ids"},
              "input_ids": render(tokenizer, row, prompt)}
             for row in [*base.cases.values(), *new]]
    io.write_jsonl(output / "cases.jsonl", cases)
    io.write_json(output / "manifest.json", {
        **{k: v for k, v in base.manifest.items() if k not in ("splits", "source")},
        "model": config["model"], "eos_token_id": tokenizer.eos_token_id,
        "base": str(config["base"]), "sources": config["sources"],
        "exclude_test": config.get("exclude_test", []), "seed": seed,
        "splits": {"train": base.ids("train") + [r["case_id"] for r in new],
                   "development": base.ids("development")}})
    return output
