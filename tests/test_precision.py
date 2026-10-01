import pytest
import torch

from crova.models import load_model


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


@pytest.fixture
def ids():
    return torch.tensor([[5, 17, 42, 8, 60, 11]])


def test_layercast_computes_fp32_with_bf16_weights(tiny_dense, ids):
    model = load_model(tiny_dense, precision="layercast", device="cpu")
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.model.norm.weight.dtype == torch.float32
    logits = _logits(model, ids)
    assert logits.dtype == torch.float32
    reference = load_model(tiny_dense, precision="fp32", device="cpu")
    # BF16-stored weights are exact in FP32, so both compute the same function.
    torch.testing.assert_close(logits, _logits(reference, ids), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("precision,dtype", [("fp16", torch.float16), ("fp32", torch.float32),
                                             ("bf16", torch.bfloat16)])
def test_casts(tiny_dense, ids, precision, dtype):
    model = load_model(tiny_dense, precision=precision, device="cpu")
    assert _logits(model, ids).dtype == dtype


@pytest.mark.parametrize("setting", ["norm", "attn", "mlp", "head", "attn+norm", "first8",
                                     "last8", "blocks", "mlp+first8", "nomlp"])
def test_selective_runs_and_returns_to_bf16(tiny_dense, ids, setting):
    model = load_model(tiny_dense, precision="selective", selective=setting, device="cpu")
    logits = _logits(model, ids)
    assert torch.isfinite(logits).all()
    if setting in ("head", "nomlp"):
        assert logits.dtype == torch.float32
    else:
        assert logits.dtype == torch.bfloat16


def test_selective_all_blocks_close_to_fp32(tiny_dense, ids):
    everything = load_model(tiny_dense, precision="selective", selective="blocks+norm+head",
                            device="cpu")
    fp32 = load_model(tiny_dense, precision="fp32", device="cpu")
    # Only the embedding output and the block outputs are rounded to BF16.
    torch.testing.assert_close(_logits(everything, ids).float(), _logits(fp32, ids),
                               atol=0.1, rtol=0.05)


def test_selective_rejects_unknown(tiny_dense):
    with pytest.raises(ValueError):
        load_model(tiny_dense, precision="selective", selective="attention", device="cpu")


def _fp32_stored_reference(model, setting):
    """The straightforward version: chosen modules converted to FP32 storage."""
    from crova.precision import _map, _to16, _to32

    parts = setting.split("+")
    for name, module in list(model.named_modules()):
        if ("attn" in parts and name.endswith(".self_attn")) or ("mlp" in parts and name.endswith(".mlp")):
            module.float()
            module.register_forward_pre_hook(
                lambda m, a, k: (_map(a, _to32), {key: _map(v, _to32) for key, v in k.items()}),
                with_kwargs=True)
            module.register_forward_hook(lambda m, a, out: _map(out, _to16))


@pytest.mark.parametrize("setting", ["attn", "mlp", "attn+mlp"])
def test_selective_keeps_bf16_weights_with_identical_outputs(tiny_dense, ids, setting):
    from crova.precision import LayerCastLinear

    model = load_model(tiny_dense, precision="selective", selective=setting, device="cpu")
    reference = load_model(tiny_dense, device="cpu")
    _fp32_stored_reference(reference, setting)
    assert torch.equal(_logits(model, ids), _logits(reference, ids))
    linears = [m for m in model.modules() if isinstance(m, (LayerCastLinear, torch.nn.Linear))]
    assert all(m.weight.dtype == torch.bfloat16 for m in linears)
