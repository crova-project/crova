"""Pinned models and model loading with a chosen precision and optional adapter."""
from __future__ import annotations

import os

import torch

from . import precision as P

# name: (Hugging Face ID, revision, number of experts selected per token or 0 for dense)
MODELS = {
    "jais": ("inceptionai/Jais-2-8B-Chat", "f7df1cb035424ed345b5ee2b04f114721c8951f2", 0),
    "olmoe": ("allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5", 8),
    "llama": ("unsloth/Llama-3.1-8B-Instruct", "4699cc75b550f9c6f3173fb80f4703b62d946aa5", 0),
    "qwenmoe": ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "ec052fda178e241c7c443468d2fa1db6618996be", 4),
    "olmo1b": ("allenai/OLMo-1B-0724-hf", "d7cbab742d80589e714b1a2d7f838dcd21cbe143", 0),
}


# Jais's published tokenizer uses a pre-tokenizer regex that transformers flags as
# incorrect; the corrected pattern changes the tokens of about a quarter of prompts.
TOKENIZER_OPTIONS = {"jais": {"fix_mistral_regex": True}}


def resolve(name):
    """Return (model ID or local path, revision, experts per token)."""
    if name in MODELS:
        return MODELS[name]
    return name, None, 0


def experts_per_token(model_config, name):
    for key in ("num_experts_per_tok", "moe_topk", "top_k"):
        value = getattr(model_config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    return resolve(name)[2]


def load_tokenizer(name):
    from transformers import AutoTokenizer

    model_id, revision, _ = resolve(name)
    return AutoTokenizer.from_pretrained(model_id, revision=revision, **TOKENIZER_OPTIONS.get(name, {}))


def resolve_device(device=None):
    """The GPU unless a config explicitly asks for `device: cpu`; never a silent CPU fallback."""
    if device in (None, "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU is visible to torch (check the driver, the torch build and "
                               "CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES); set `device: cpu` to "
                               "run on CPU deliberately")
        return "cuda"
    return device


def numerics():
    """Ordinary nondeterministic execution without TF32 shortcuts."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(name, *, precision="bf16", load="cast", selective=None, adapter=None,
               device=None):
    """Load a causal LM for inference.

    precision: bf16 (native), fp16, fp32, layercast or selective.
    load: "cast" loads the BF16 checkpoint and converts it; "direct" asks
      transformers to load in the target dtype.
    selective: module selection for precision="selective" (see precision.py).
    adapter: optional PEFT output-head LoRA directory.
    device: "cuda" (default; fails if no GPU is visible) or "cpu".
    """
    from transformers import AutoModelForCausalLM

    numerics()
    model_id, revision, _ = resolve(name)
    dtype = P.DTYPES.get(precision, torch.bfloat16) if load == "direct" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="eager")
    model.eval().requires_grad_(False)
    P.apply(model, precision, selective=selective)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()
    return model.to(resolve_device(device))


def environment():
    import transformers

    record = {"torch": torch.__version__, "transformers": transformers.__version__,
              "cuda": torch.version.cuda, "hip": torch.version.hip,
              "host": os.uname().nodename}
    if torch.cuda.is_available():
        record["device"] = torch.cuda.get_device_name(0)
    return record
