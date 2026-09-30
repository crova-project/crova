"""Precision interventions applied to a loaded BF16 model.

bf16       native execution (no change)
fp16       every parameter cast to FP16
fp32       every parameter cast to FP32
layercast  FP32 computation everywhere; linear weights stay stored in BF16 and
           are upcast inside each matmul (FP32 upcasting)
selective  upcast only chosen parts to FP32, then return to BF16 at their output.
           Parts are joined with "+": norm, attn, mlp, head, and at most one block
           range: first8, last8 or blocks (all layers). "nomlp" runs everything in
           FP32 except the MLPs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
          "layercast": torch.float32}
PARTS = ("norm", "attn", "mlp", "head")
BLOCKS = ("first8", "last8", "blocks")


class LayerCastLinear(torch.nn.Module):
    """BF16 weight storage; FP32 input, multiplication and output."""

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = torch.nn.Parameter(linear.weight.detach().to(torch.bfloat16),
                                         requires_grad=False)
        self.bias = (torch.nn.Parameter(linear.bias.detach().to(torch.bfloat16),
                                        requires_grad=False)
                     if linear.bias is not None else None)

    def forward(self, x):
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.weight.float(), bias)


def adapt_layercast(model):
    model.float()
    replaced = []
    for name, module in list(model.named_modules()):
        if type(module) is torch.nn.Linear:
            parent, _, child = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, child, LayerCastLinear(module))
            replaced.append(name)
    if "lm_head" not in replaced:
        raise ValueError("LayerCast expects an lm_head linear layer")
    return replaced


def _map(x, fn):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, tuple):
        return tuple(_map(v, fn) for v in x)
    if isinstance(x, list):
        return [_map(v, fn) for v in x]
    return x


def _to32(t):
    return t.float() if t.is_floating_point() else t


def _to16(t):
    return t.to(torch.bfloat16) if t.dtype == torch.float32 else t


def _upcast(module, *, keep_output=False):
    module.float()
    module.register_forward_pre_hook(
        lambda m, args, kwargs: (_map(args, _to32), {k: _map(v, _to32) for k, v in kwargs.items()}),
        with_kwargs=True)
    if not keep_output:
        module.register_forward_hook(lambda m, args, out: _map(out, _to16))


def _layers(model):
    return model.model.layers


def adapt_selective(model, setting):
    parts = setting.split("+")
    unknown = set(parts) - set(PARTS) - set(BLOCKS) - {"nomlp"}
    if unknown or not parts:
        raise ValueError(f"unknown selective FP32 setting: {setting}")
    chosen = []
    if parts == ["nomlp"]:
        model.float()
        for name, module in model.named_modules():
            if name.endswith(".mlp"):
                module.to(torch.bfloat16)
                module.register_forward_pre_hook(
                    lambda m, args, kwargs: (_map(args, lambda t: t.to(torch.bfloat16)
                                                   if t.is_floating_point() else t), kwargs),
                    with_kwargs=True)
                module.register_forward_hook(lambda m, args, out: _map(out, _to32))
                chosen.append(name)
        return chosen
    count = len(_layers(model))
    blocks = [p for p in parts if p in BLOCKS]
    if len(blocks) > 1:
        raise ValueError("choose at most one of first8, last8, blocks")
    whole = []
    if blocks:
        whole = {"first8": range(8), "last8": range(count - 8, count), "blocks": range(count)}[blocks[0]]
    covered = tuple(f"model.layers.{i}." for i in whole)
    for name, module in model.named_modules():
        pick = (("norm" in parts and "Norm" in type(module).__name__)
                or ("attn" in parts and name.endswith(".self_attn"))
                or ("mlp" in parts and name.endswith(".mlp"))
                or ("head" in parts and name == "lm_head"))
        if pick and not name.startswith(covered):
            _upcast(module, keep_output=name == "lm_head")
            chosen.append(name)
    for i in whole:
        _upcast(_layers(model)[i])
        chosen.append(f"model.layers.{i}")
    if not chosen:
        raise ValueError(f"selective FP32 setting {setting} selected nothing")
    return chosen


def apply(model, precision, *, selective=None):
    if precision == "bf16":
        return []
    if precision in ("fp16", "fp32"):
        model.to(DTYPES[precision])
        return []
    if precision == "layercast":
        return adapt_layercast(model)
    if precision == "selective":
        if not selective:
            raise ValueError("precision=selective needs a 'selective' setting, e.g. attn+norm")
        return adapt_selective(model, selective)
    raise ValueError(f"unknown precision: {precision}")
