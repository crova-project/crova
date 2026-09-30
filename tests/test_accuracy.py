import torch

from crova import accuracy, io
from crova.models import load_model


class StubTokenizer:
    eos_token_id = 3

    def decode(self, ids, skip_special_tokens=True):
        return " The answer is C." if len(ids) else ""


def test_letter_scoring_picks_highest_letter_logit(tiny_dense):
    model = load_model(tiny_dense, device="cpu")
    row = {"choices": ["w", "x", "y"], "answer": "A"}
    letters = [10, 11, 12, 13, 14]
    ids = [5, 6, 7]
    predicted, finite = accuracy.score(model, None, row, ids, "plain", letters=letters)
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
    assert predicted == "ABC"[int(logits[letters[:3]].argmax())] and finite


def test_chat_scoring_parses_generated_letter(tiny_dense):
    model = load_model(tiny_dense, device="cpu")
    predicted, _ = accuracy.score(model, StubTokenizer(), {"choices": ["a", "b", "c", "d"]},
                                  [5, 6, 7], "chat")
    assert predicted == "C"


def test_parser():
    assert accuracy.CHOICE.search("B").group() == "B"
    assert accuracy.CHOICE.search(" (C) because").group() == "C"
    assert accuracy.CHOICE.search("ABC") is None


def test_summary_excludes_workload_questions(tmp_path):
    rows = [{"case_id": f"mmlu-{i}", "category": "mmlu", "predicted": "A", "answer": "A" if i else "B",
             "correct": bool(i), "parsed": True, "finite": True} for i in range(3)]
    io.write_jsonl(tmp_path / "w" / "cases.jsonl", [{"case_id": "mmlu-0"}])
    io.write_json(tmp_path / "w" / "manifest.json", {"splits": {}})
    summary = accuracy.summarize(rows, tmp_path / "w")["mmlu"]
    assert summary["all"]["correct"] == 2 and summary["all"]["cases"] == 3
    assert summary["excluding_workload"] == {"cases": 2, "correct": 2, "accuracy": 1.0,
                                             "unparsed": 0, "nonfinite": 0}
    io.write_json(tmp_path / "a.json", {"rows": rows[:2]})
    io.write_json(tmp_path / "b.json", {"rows": rows[2:]})
    merged = accuracy.merge([tmp_path / "a.json", tmp_path / "b.json"], tmp_path / "m.json")
    assert merged["mmlu"]["all"]["cases"] == 3
