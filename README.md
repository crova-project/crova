# CroVA

Code for *Same Model, Different Numbers: Measuring and Reducing Cross-Vendor
Mismatch in Language Models*.

The same model with the same weights produces different logits on NVIDIA and AMD
GPUs. This repository measures that mismatch against a reference GPU and tests
ways to reduce it on a target GPU: higher-precision execution (FP16, FP32
upcasting, selective FP32) and an output-head LoRA trained to match the
reference logits. It also measures downstream effects on accuracy, cost and
knowledge distillation.

Supported models: Jais-2-8B-Chat, OLMoE-1B-7B, Llama-3.1-8B-Instruct and
Qwen1.5-MoE-A2.7B-Chat (pinned revisions in `src/crova/models.py`), plus
OLMo-1B as a distillation student. Any Hugging Face causal LM path also works.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv && uv sync --extra nvidia   # NVIDIA GPU (torch 2.7.1+cu128)
uv venv && uv sync --extra amd      # AMD GPU (torch 2.7.1+rocm6.3)
uv venv && uv sync --extra cpu --extra test && uv run pytest   # CPU tests
```

Add `--extra wandb` to log LoRA training scalars to Weights & Biases (set
`wandb: true` in the config; project and entity come from `WANDB_PROJECT` and
`WANDB_ENTITY`).

## Usage

Every stage is one command driven by a YAML file; `--set key=value` overrides
any field. Example configs for each model are in `configs/<model>/`. The steps
below use Jais; the other models are identical with their own configs.

A run has two sides. The **reference** GPU (NVIDIA in the paper) produces
responses and logits once. The **target** GPU (AMD) reruns the same inputs under
each intervention and is compared against the reference. Copy the reference
outputs (`runs/<model>/nvidia/`) to the target machine between the two steps.

### 1. Workload

Select train and development questions from the benchmarks and tokenize them:

```bash
crova workload --config configs/jais/workload.yaml
```

Jais uses MMLU and ArabicMMLU with a chat prompt; the other models use MMLU and
ARC-Challenge with a plain `Answer:` prompt, drawn from non-test splits.

### 2. Reference GPU

```bash
crova generate --config configs/jais/generate.yaml           # greedy responses
crova forward  --config configs/jais/forward-reference.yaml  # logits at every response position
```

`forward` runs one teacher-forced pass over prompt + response and stores the
logits (and router logits for MoE models) per case. Both commands accept
`--set shard=i --set num_shards=n` to split the work across GPUs.

### 3. Target GPU: measure the mismatch

```bash
crova forward --config configs/jais/forward-target.yaml                 # per-case statistics
crova generate --config configs/jais/generate.yaml \
  --set splits=[development] --set output=runs/jais/amd/bf16/responses   # whole-response agreement
crova compare --config configs/jais/compare.yaml                        # pooled metrics + 95% CIs
```

Reported metrics: raw-logit MSE, probability MSE, forward KL, top-1, ordered
top-5 and top-5 set agreement, bit-identical BF16 logits, whole-response
agreement, and for MoE models expert-selection agreement per layer and position.

Compare two targets question by question (paired bootstrap):

```bash
crova bootstrap runs/jais/amd/bf16/summary.json runs/jais/amd/fp32/summary.json
```

### 4. Precision interventions

Set `precision` in any model-loading config:

| precision | meaning |
|---|---|
| `bf16` | native execution |
| `fp16` | all weights and computation in FP16 |
| `fp32` | all weights and computation in FP32 |
| `layercast` | FP32 computation, linear weights kept in BF16 and upcast per matmul |
| `selective` | FP32 only for the parts in `selective`, e.g. `attn+norm`, `mlp`, `head`, `last8`, `first8`, `blocks`, `nomlp` |

`load: direct` loads the checkpoint straight into the target dtype instead of
casting the BF16 model. With FP16/FP32 targets, `round_to_bf16: true` also
records statistics after rounding the logits to BF16.

```bash
crova forward --config configs/jais/forward-target.yaml \
  --set precision=selective --set selective=attn+norm --set output=runs/jais/amd/attn-norm/stats
```

### 5. Output-head LoRA

```bash
crova train-lora --config configs/jais/lora.yaml --set profile=kl --set output=runs/jais/lora/kl
crova forward --config configs/jais/forward-target.yaml \
  --set adapter=runs/jais/lora/kl/adapter --set output=runs/jais/amd/lora-kl/stats
```

The frozen BF16 model gets a rank-32 LoRA on its output head, trained on the
target GPU towards the reference logits of the training responses. Four losses
are combined: raw-logit MSE, forward KL, and two ranking losses that preserve
the reference gaps around the top-1 and top-5 tokens (`src/crova/losses.py`).
Each is divided by its initial mean before weighting. Profiles:

| profile | MSE | KL | token gap | top-5 gap |
|---|---|---|---|---|
| `equal` | 0.25 | 0.25 | 0.25 | 0.25 |
| `numerical` | 0.4 | 0.4 | 0.1 | 0.1 |
| `mse` | 0.7 | 0.1 | 0.1 | 0.1 |
| `kl` | 0.1 | 0.7 | 0.1 | 0.1 |
| `token` | 0.1 | 0.1 | 0.7 | 0.1 |
| `top5` | 0.1 | 0.1 | 0.1 | 0.7 |
| `decision` | 0.1 | 0.1 | 0.4 | 0.4 |
| `kl_only` | 0 | 1 | 0 | 0 |

Training uses AdamW with step-size backtracking (see `src/crova/lora.py`). The
saved adapter is an ordinary PEFT adapter and loads with `adapter:` in any
config.

`crova head-capacity` reports how much of the training-set mismatch any linear
map of the output-head input could remove (rank-limited and unrestricted).

### 6. Accuracy and cost

```bash
crova accuracy --config configs/jais/accuracy.yaml
crova cost     --config configs/jais/cost.yaml
```

Accuracy covers the complete test sets. Jais generates up to 8 tokens after a
chat prompt and the first standalone letter is parsed; the other models are
scored by the highest answer-letter logit after `Answer:`. `cost` records
runtime and peak GPU memory for the full benchmark and the development forwards.

### 7. Distillation

A student (OLMo-1B) is distilled from OLMoE teachers that ran on different GPUs
or precisions; only the teacher differs between students.

```bash
crova forward  --config configs/kd/teacher.yaml       # top-64 teacher targets, once per teacher
crova kd-train --config configs/kd/train.yaml         # one student per teacher (and seed)
crova kd-eval  --config configs/kd/evaluate.yaml      # KL and top-1 agreement with the reference student
crova accuracy --config configs/kd/accuracy.yaml
crova kd-agreement runs/kd/students/a/accuracy.json runs/kd/students/b/accuracy.json
crova kd-targets runs/kd/teachers/nvidia-bf16 runs/kd/teachers/amd-fp32 --workload runs/olmoe/workload
```

### 8. Matmul case study

One projection computed with the vendor matmul, a fixed-order tensor-core Triton
kernel and a sequential FP32 Triton kernel, on each GPU:

```bash
crova matmul-capture --config configs/matmul/capture.yaml --set output=runs/matmul/nvidia
crova matmul-capture --config configs/matmul/capture.yaml --set output=runs/matmul/amd
crova matmul-compare runs/matmul/nvidia runs/matmul/amd --output runs/matmul/compare.json
crova matmul-inspect --config configs/matmul/inspect.yaml   # MFMA/MMA instructions in the compiled kernel
```

The operands file is a safetensors file with BF16 tensors `weight` [N, K] and
one or more inputs [..., K]. The paper uses the input and weight of the first
`q_proj` of SmolLM2-135M for a 27-token prompt (M=27, N=K=576).

## Output layout

| path | contents |
|---|---|
| `workload/manifest.json`, `cases.jsonl` | splits, questions and prompt token IDs |
| `responses/<case>.json` | greedy response token IDs |
| `logits/<case>.safetensors` | `logits` [positions, vocab], `response`, optional `router_logits`, `hidden` |
| `stats/<case>.stats.json` | per-case sums used for pooled metrics and bootstrap intervals |
| `lora/<profile>/` | `normalizers.json`, `progress.jsonl`, `adapter/` |

## Tests

```bash
uv run pytest
```

The tests use tiny random models on CPU and need no downloads.

## License

Apache-2.0. See `LICENSE`.
