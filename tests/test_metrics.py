import torch

from crova.metrics import case_stats, paired_bootstrap, rollout_stats, summarize


def _pair(noise, seed=0, m=5, vocab=30):
    g = torch.Generator().manual_seed(seed)
    reference = torch.randn(m, vocab, generator=g).to(torch.bfloat16)
    target = (reference.float() + noise * torch.randn(m, vocab, generator=g)).to(torch.bfloat16)
    response = torch.arange(m, dtype=torch.int32)
    return {"logits": reference, "response": response}, {"logits": target, "response": response}


def test_identical_logits():
    reference, _ = _pair(0.0)
    s = case_stats(reference, reference)
    assert s["sq_err"] == 0 and abs(s["kl"]) < 1e-6
    assert s["top1"] == s["ord5"] == s["set5"] == 5
    assert s["equal_bits"] == 5 * 30


def test_values_match_direct_formulas():
    reference, target = _pair(0.5)
    s = case_stats(reference, target)
    a, b = reference["logits"].float(), target["logits"].float()
    assert abs(s["sq_err"] - float((a - b).square().sum())) < 1e-3
    kl = (a.softmax(-1) * (a.log_softmax(-1) - b.log_softmax(-1))).sum()
    assert abs(s["kl"] - float(kl)) < 1e-4
    assert s["top1"] == int((a.argmax(-1) == b.argmax(-1)).sum())


def test_router_statistics():
    reference, target = _pair(0.0)
    router = torch.randn(2, 5, 8)
    reference["router_logits"], target["router_logits"] = router, router.clone()
    target["router_logits"][0, 0] = -router[0, 0]
    s = case_stats(reference, target, experts_per_token=2)
    assert s["router_slots"] == 10 and s["router_set_equal"] == 9
    assert s["router_positions_all_layers_equal"] == 4


def test_summary_and_bootstrap():
    per_a = {f"c{i}": case_stats(*_pair(0.5, seed=i)) for i in range(8)}
    per_b = {f"c{i}": case_stats(*_pair(0.2, seed=i)) for i in range(8)}
    summary = summarize(per_a, {"c0": rollout_stats([1, 2], [1, 2])}, resamples=200)
    assert summary["cases"] == 8 and summary["positions"] == 40
    low, high = summary["raw_logit_mse_ci95"]
    assert low <= summary["raw_logit_mse"] <= high
    assert summary["rollout_equal"] == 1
    result = paired_bootstrap(per_a, per_b, resamples=200)
    assert result["sq_err_rel_change_pct"][0] < 0  # smaller noise, smaller error


def test_rollout_first_divergence():
    assert rollout_stats([1, 2, 3], [1, 5, 3]) == {
        "equal": False, "first_divergence": 1, "reference_length": 3, "target_length": 3}
