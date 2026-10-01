"""End-to-end smoke test of every stage on tiny random models (CPU)."""
import pytest
import torch
from conftest import make_workload
from safetensors.torch import load_file

from crova import capture, head_capacity, io, kd, lora


@pytest.fixture(params=[("dense", "teacher_forced"), ("moe", "teacher_forced"), ("dense", "prefix"),
                        ("moe", "prefix")], ids=lambda p: "-".join(p))
def setup(request, tmp_path, tiny_dense, tiny_moe):
    kind, mode = request.param
    model = tiny_dense if kind == "dense" else tiny_moe
    workload = make_workload(tmp_path / "workload")
    base = {"model": model, "workload": str(workload), "splits": ["train", "development"],
            "mode": mode}
    capture.generate({**base, "output": str(tmp_path / "responses"), "max_new_tokens": 6})
    capture.forward({**base, "responses": str(tmp_path / "responses"),
                     "output": str(tmp_path / "reference"), "save_hidden": True})
    return model, workload, base, tmp_path


def test_generate_and_forward(setup):
    model, workload, base, tmp = setup
    response = io.read_json(io.case_file(tmp / "responses", "mmlu-0000", ".json"))["response"]
    values = load_file(str(io.case_file(tmp / "reference", "mmlu-0000", ".safetensors")))
    assert values["logits"].shape == (len(response), 97)
    assert values["logits"].dtype == torch.bfloat16
    assert values["hidden"].shape == (len(response), 32)
    assert ("router_logits" in values) == ("moe" in model)
    # The captured logits reproduce the greedy tokens (same device, precision and mode).
    assert values["logits"].float().argmax(-1).tolist() == response
    if "router_logits" in values:
        assert values["router_logits"].shape == (2, len(response), 8)


def test_modes_compute_the_same_function(tiny_dense, tmp_path):
    from crova.models import load_model

    model = load_model(tiny_dense, precision="fp32", device="cpu")
    prompt, response = [5, 9, 13, 40], [7, 22, 81, 3]
    with torch.no_grad():
        a = capture.position_logits(model, prompt, response, "teacher_forced", save_hidden=True)
        b = capture.position_logits(model, prompt, response, "prefix", save_hidden=True)
        assert capture.greedy(model, prompt, 3, 6, "teacher_forced") == capture.greedy(
            model, prompt, 3, 6, "prefix")
    torch.testing.assert_close(a["logits"], b["logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a["hidden"], b["hidden"], atol=1e-5, rtol=1e-5)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        capture.mode_of({"mode": "cached"})


def test_target_statistics_and_compare(setup):
    model, workload, base, tmp = setup
    capture.forward({**base, "responses": str(tmp / "responses"), "precision": "fp32",
                     "reference": str(tmp / "reference"), "output": str(tmp / "fp32"),
                     "round_to_bf16": True})
    capture.generate({**base, "precision": "fp32", "output": str(tmp / "fp32-responses"),
                      "max_new_tokens": 6})
    summary = capture.compare({"workload": str(workload), "stats": str(tmp / "fp32"),
                               "reference_responses": str(tmp / "responses"),
                               "target_responses": str(tmp / "fp32-responses"),
                               "output": str(tmp / "fp32.json")})
    assert summary["cases"] == 2 and summary["raw_logit_mse"] > 0
    assert "equal_bits" not in summary  # FP32 target: bitwise equality is not defined
    rounded = capture.compare({"workload": str(workload), "stats": str(tmp / "fp32"),
                               "use_rounded_bf16": True, "output": str(tmp / "rounded.json")})
    assert "equal_bits" in rounded
    same = capture.compare({"workload": str(workload), "reference": str(tmp / "reference"),
                            "target": str(tmp / "reference"), "output": str(tmp / "same.json")})
    assert same["raw_logit_mse"] == 0 and same["equal_bits_pct"] == 100


def test_lora_learns_a_known_head_shift(setup, tmp_path):
    """Target = the model's own logits plus a fixed low-rank shift; LoRA should reduce every loss."""
    model, workload, base, tmp = setup
    shifted = tmp / "shifted"
    shifted.mkdir()
    g = torch.Generator().manual_seed(3)
    direction = torch.randn(32, 97, generator=g) * 0.5
    for cid in io.read_json(workload / "manifest.json")["splits"]["train"]:
        values = load_file(str(io.case_file(tmp / "reference", cid, ".safetensors")))
        values["logits"] = (values["logits"].float() + values["hidden"].float() @ direction
                            ).to(torch.bfloat16)
        from safetensors.torch import save_file

        save_file(values, str(io.case_file(shifted, cid, ".safetensors")))
    out = lora.train({"model": model, "mode": base["mode"], "workload": str(workload),
                      "reference": str(shifted),
                      "output": str(tmp_path / "lora"), "profile": "equal", "epochs": 30,
                      "lr": 1e-2, "save_steps": [4]})
    progress = io.read_jsonl(out / "progress.jsonl")
    assert len(progress) == 120
    first, last = progress[0]["losses"], progress[-1]["losses"]
    assert all(last[k] < first[k] for k in first)
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "step-0004").exists()
    # The saved adapter loads for inference through the ordinary model path.
    capture.forward({**base, "adapter": str(out / "adapter"), "responses": str(tmp / "responses"),
                     "reference": str(tmp / "reference"), "output": str(tmp_path / "adapted")})


def test_backtracking_restores_rejected_updates():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([p], lr=10.0)

    def evaluate(backward):  # any move increases this objective
        value = (p - torch.tensor([1.0, -2.0])).square().sum() + 1.0
        if backward:
            (value + 0.0 * p.sum()).backward()
            p.grad = torch.tensor([1.0, 1.0])  # misleading gradient
        return float(value.detach()), {}

    record = lora.backtracking_step([p], optimizer, evaluate)
    assert record["fraction"] == 0.0
    assert p.detach().tolist() == [1.0, -2.0]
    assert optimizer.state_dict()["state"] == {}


def test_schedule_is_deterministic():
    assert lora.schedule(5, 2) == lora.schedule(5, 2)
    assert sorted(lora.schedule(5, 2)) == sorted(list(range(5)) * 2)


def test_head_capacity(setup):
    model, workload, base, tmp = setup
    capture.forward({**base, "splits": ["train"], "precision": "fp16", "save_hidden": True,
                     "responses": str(tmp / "responses"), "output": str(tmp / "fp16")})
    result = head_capacity.run({"workload": str(workload), "target": str(tmp / "fp16"),
                                "reference": str(tmp / "reference"), "rank": 4,
                                "output": str(tmp / "capacity.json")})
    assert 0 <= result["rank_r_fraction"] <= result["any_rank_fraction"] <= 1 + 1e-4
    # Synthetic check: a residual that is exactly a rank-2 map of H is fully captured.
    h = torch.randn(40, 8)
    e = h @ torch.randn(8, 2) @ torch.randn(2, 30)
    result = head_capacity.capacity([h[:25], h[25:]], [e[:25], e[25:]], rank=2)
    assert abs(result["rank_r_fraction"] - 1) < 1e-4
    noise = head_capacity.capacity([h], [torch.randn(40, 30)], rank=2)
    assert 0 < noise["rank_r_fraction"] < noise["any_rank_fraction"] < 1


def test_distillation_targets(setup):
    model, workload, base, tmp = setup
    kd.topk({"teacher": str(tmp / "reference"), "workload": str(workload),
             "output": str(tmp / "topk"), "k": 8})
    values = load_file(str(io.case_file(tmp / "topk", "mmlu-0000", ".safetensors")))
    assert values["topk_logprob"].shape[-1] == 8
    same = kd.compare_targets(tmp / "topk", tmp / "topk", workload)
    assert same["same_order_pct"] == 100 and same["mean_total_variation"] < 1e-6
    capture.forward({**base, "splits": ["train"], "responses": str(tmp / "responses"),
                     "output": str(tmp / "direct-topk"), "topk": 8})
    direct = load_file(str(io.case_file(tmp / "direct-topk", "mmlu-0000", ".safetensors")))
    assert torch.equal(direct["topk_index"], values["topk_index"])
    summary = kd.evaluate({"students": {"a": model, "b": model}, "workload": str(workload),
                           "responses": str(tmp / "responses"), "output": str(tmp / "kd.json")})
    assert summary["b"]["top1_agreement_pct"] == 100


def test_lora_rejects_a_target_that_already_matches(setup, tmp_path):
    model, workload, base, tmp = setup
    with pytest.raises(ValueError, match="normalizers"):
        lora.train({"model": model, "mode": base["mode"], "workload": str(workload),
                    "reference": str(tmp / "reference"), "output": str(tmp_path / "same"),
                    "profile": "equal", "epochs": 1})
