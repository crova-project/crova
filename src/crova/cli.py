"""Command-line entry point: crova <command> --config file.yaml [--set key=value ...]."""
from __future__ import annotations

import argparse
import json

import yaml

from . import io

CONFIG_COMMANDS = {
    "workload": ("crova.workload", "build", "select and tokenize train/development questions"),
    "generate": ("crova.capture", "generate", "greedy responses"),
    "forward": ("crova.capture", "forward", "teacher-forced logits over given responses"),
    "compare": ("crova.capture", "compare", "pool reference-versus-target metrics"),
    "train-lora": ("crova.lora", "train", "train the output-head LoRA"),
    "accuracy": ("crova.accuracy", "evaluate", "multiple-choice accuracy on full test sets"),
    "cost": ("crova.cost", "measure", "runtime and peak GPU memory"),
    "head-capacity": ("crova.head_capacity", "run", "linear head-correction capacity"),
    "kd-topk": ("crova.kd", "topk", "reduce teacher logits to top-K targets"),
    "kd-train": ("crova.kd", "train", "distil a student from one teacher"),
    "kd-eval": ("crova.kd", "evaluate", "compare students with a reference student"),
    "matmul-capture": ("crova.matmul.probe", "capture", "matmul case study outputs"),
    "matmul-inspect": ("crova.matmul.probe", "inspect", "compiled kernel instructions"),
}


def _override(config, assignments):
    for item in assignments or []:
        key, _, value = item.partition("=")
        config[key] = yaml.safe_load(value)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crova")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, help_text) in CONFIG_COMMANDS.items():
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True)
        sub.add_argument("--set", action="append", metavar="KEY=VALUE",
                         help="override a config value (YAML syntax)")
    sub = commands.add_parser("bootstrap", help="paired bootstrap of two compare outputs")
    sub.add_argument("baseline"), sub.add_argument("candidate"), sub.add_argument("--output")
    sub = commands.add_parser("accuracy-merge", help="combine sharded accuracy outputs")
    sub.add_argument("inputs", nargs="+"), sub.add_argument("--output", required=True)
    sub.add_argument("--exclude-workload")
    sub = commands.add_parser("kd-agreement", help="same-answer rate of two accuracy outputs")
    sub.add_argument("reference"), sub.add_argument("other")
    sub = commands.add_parser("kd-targets", help="compare two teachers' top-K targets")
    sub.add_argument("reference"), sub.add_argument("other")
    sub.add_argument("--workload", required=True)
    sub = commands.add_parser("matmul-compare", help="compare two matmul captures")
    sub.add_argument("first"), sub.add_argument("second"), sub.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command in CONFIG_COMMANDS:
        import importlib

        module, function, _ = CONFIG_COMMANDS[args.command]
        config = _override(io.load_config(args.config), args.set)
        result = getattr(importlib.import_module(module), function)(config)
    elif args.command == "bootstrap":
        from .metrics import paired_bootstrap

        result = paired_bootstrap(io.read_json(args.baseline)["per_case"],
                                  io.read_json(args.candidate)["per_case"])
        if args.output:
            io.write_json(args.output, result)
    elif args.command == "accuracy-merge":
        from .accuracy import merge

        result = merge(args.inputs, args.output, args.exclude_workload)
    elif args.command == "kd-agreement":
        from .kd import agreement

        result = agreement(args.reference, args.other)
    elif args.command == "kd-targets":
        from .kd import compare_targets

        result = compare_targets(args.reference, args.other, args.workload)
    else:
        from .matmul.probe import compare

        result = compare(args.first, args.second, args.output)
    if isinstance(result, dict):
        print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
