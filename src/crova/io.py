"""Small file helpers shared by every stage."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return config


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def case_file(directory, case_id, suffix):
    """Per-case file name; case IDs may contain '/'."""
    return Path(directory) / (case_id.replace("/", "__") + suffix)
