"""Matmul case study: one projection computed three ways on one GPU.

The operands file (safetensors) holds BF16 tensors
  weight            [N, K]   projection weight
  <input name>      [..., K] one or more inputs, e.g. the input of SmolLM2-135M's
                             first q_proj for one prompt (the paper uses a
                             27-token prompt: M=27, N=K=576)
Every input tensor is flattened to [M, K] and multiplied by weight.T with
  vendor        torch's default matmul (vendor BLAS)
  ordered_tile  Triton tiles with a fixed K order through tensor-core dot
  scalar        Triton sequential FP32 multiply-add over K
each producing BF16 and FP32 outputs. Run `capture` on each GPU, then `compare`.
`inspect` saves the compiled ordered-tile kernel and counts its matrix-core
instructions (MFMA on AMD, MMA on NVIDIA).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .. import io
from ..models import environment
from .kernels import ordered_tile, scalar

METHODS = ("vendor", "ordered_tile", "scalar")


def _operation(method, left, weight, dtype):
    m, k = left.shape
    n = weight.shape[0]
    if method == "vendor":
        if dtype == torch.bfloat16:
            return lambda: torch.nn.functional.linear(left, weight)
        return lambda: torch.mm(left, weight.T, out_dtype=torch.float32)
    factory = ordered_tile if method == "ordered_tile" else scalar
    return factory(left, weight, m, n, k, dtype)


def _inputs(operands):
    weight = operands.pop("weight").contiguous()
    return weight, {name: t.reshape(-1, weight.shape[1]).contiguous() for name, t in operands.items()}


def capture(config):
    """config keys: operands, output."""
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    weight, inputs = _inputs(load_file(config["operands"], device="cuda"))
    outputs, report = {}, {"environment": environment(), "cases": {}}
    for name, left in inputs.items():
        for method in METHODS:
            for label, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                key = f"{name}.{method}.{label}"
                operation = _operation(method, left, weight, dtype)
                value = operation()
                torch.cuda.synchronize()
                repeat = all(torch.equal(value, operation()) for _ in range(3))
                outputs[key] = value.cpu().contiguous()
                report["cases"][key] = {"shape": list(value.shape), "repeat_equal": repeat}
    save_file(outputs, str(out / "outputs.safetensors"))
    io.write_json(out / "report.json", report)


def compare(first, second, output):
    """Element-wise disagreement between two captures (e.g. two GPUs)."""
    a = load_file(str(Path(first) / "outputs.safetensors"))
    b = load_file(str(Path(second) / "outputs.safetensors"))
    result = {}
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        bits = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
        result[key] = {"elements": x.numel(),
                       "unequal": int((x.view(bits) != y.view(bits)).sum()),
                       "max_abs_difference": float((x.float() - y.float()).abs().max())}
    io.write_json(output, result)
    return result


INSTRUCTIONS = {"amdgcn": r"\bv_mfma_[A-Za-z0-9_]+", "ptx": r"\bmma\.sync[.\w]*|\bwgmma[.\w]*"}


def inspect(config):
    """config keys: operands, output. Saves compiler stages of the FP32 ordered-tile kernel."""
    import inspect as pyinspect

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    weight, inputs = _inputs(load_file(config["operands"], device="cuda"))
    left = next(iter(inputs.values()))
    m, k = left.shape
    n = weight.shape[0]
    operation = ordered_tile(left, weight, m, n, k, torch.float32)
    kernel = pyinspect.getclosurevars(operation).nonlocals["kernel"]
    result = torch.empty((m, n), device="cuda", dtype=torch.float32)
    compiled = kernel[((m + 31) // 32, (n + 63) // 64)](
        left, weight, result, M=m, N=n, K=k, BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    counts = {}
    for stage, content in compiled.asm.items():
        path = out / f"kernel.{stage}"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
            if stage in INSTRUCTIONS:
                counts[stage] = dict(Counter(re.findall(INSTRUCTIONS[stage], content)))
    io.write_json(out / "instructions.json", {"environment": environment(), "static_counts": counts})
    return counts
