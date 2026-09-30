"""Tiny random models saved locally, so tests run on CPU without downloads."""
import pytest
import torch


def _save(tmp_path_factory, name, config):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    path = tmp_path_factory.mktemp(name)
    model.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_dense(tmp_path_factory):
    from transformers import LlamaConfig

    return _save(tmp_path_factory, "dense", LlamaConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=10,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=3, bos_token_id=1, pad_token_id=0))


@pytest.fixture(scope="session")
def tiny_moe(tmp_path_factory):
    from transformers import OlmoeConfig

    return _save(tmp_path_factory, "moe", OlmoeConfig(
        vocab_size=97, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
        max_position_embeddings=128, eos_token_id=3, bos_token_id=1, pad_token_id=0))


def make_workload(path, *, cases=6, prompt_length=5, eos=3, seed=0):
    """A workload directory without a tokenizer: random prompts, 4 train / 2 dev."""
    from crova import io

    generator = torch.Generator().manual_seed(seed)
    rows = [{"case_id": f"mmlu-{i:04d}", "category": "mmlu", "subject": f"s{i % 2}",
             "question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": "A",
             "input_ids": torch.randint(5, 97, (prompt_length,), generator=generator).tolist()}
            for i in range(cases)]
    io.write_jsonl(path / "cases.jsonl", rows)
    io.write_json(path / "manifest.json", {
        "model": "tiny", "prompt": "plain", "eos_token_id": eos,
        "splits": {"train": [r["case_id"] for r in rows[:4]],
                   "development": [r["case_id"] for r in rows[4:]]}})
    return path
