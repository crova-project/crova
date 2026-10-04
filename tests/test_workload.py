from crova import workload


class StubTokenizer:
    eos_token_id = 0

    def __call__(self, text):
        return type("Encoded", (), {"input_ids": [ord(c) for c in text]})

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": [len(messages[0]["content"])]}


ROW = {"question": "2+2?", "choices": ["3", "4"], "answer": "B"}


def test_plain_prompt():
    ids = workload.render(StubTokenizer(), ROW, "plain")
    assert "".join(map(chr, ids)) == "2+2?\nA. 3\nB. 4\nAnswer:"


def test_chat_prompt_contains_instruction():
    ids = workload.render(StubTokenizer(), ROW, "chat")
    assert ids == [len("2+2?\nA. 3\nB. 4\n" + workload.CHAT_INSTRUCTION)]


def test_normalize_benchmarks():
    arabic = workload._normalize("arabicmmlu", {"Subject": "s", "Question": "q", "Answer Key": "B",
                                                "Option 1": "x", "Option 2": "y", "Option 3": None,
                                                "Option 4": None, "Option 5": None})
    assert arabic["choices"] == ["x", "y"] and arabic["answer"] == "B"
    arc = workload._normalize("arc", {"question": "q", "answerKey": "2",
                                      "choices": {"label": ["1", "2", "3"], "text": ["a", "b", "c"]}})
    assert arc["answer"] == "B"
    mmlu = workload._normalize("mmlu", {"subject": "s", "question": "q", "choices": ["a", "b"],
                                        "answer": 0})
    assert mmlu["answer"] == "A" and mmlu["case_id"].startswith("mmlu-")


def test_round_robin_balances_subjects_and_is_deterministic():
    rows = [{"case_id": f"c{i}", "subject": f"s{i % 3}"} for i in range(30)]
    first = workload._round_robin(rows, 6, "seed", "train")
    assert first == workload._round_robin(list(reversed(rows)), 6, "seed", "train")
    assert sorted(r["subject"] for r in first) == ["s0", "s0", "s1", "s1", "s2", "s2"]


def test_extend_keeps_base_and_avoids_test_overlap(tmp_path, monkeypatch):
    from crova import io

    def row(category, i):
        return {"case_id": f"{category}-{i}", "category": category, "subject": "s",
                "question": f"q{i}", "choices": ["a", "b"], "answer": "A"}

    base = tmp_path / "base"
    io.write_jsonl(base / "cases.jsonl", [{**row("mmlu", 0), "input_ids": [1]},
                                          {**row("mmlu", 1), "input_ids": [2]}])
    io.write_json(base / "manifest.json", {"model": "x", "prompt": "plain", "eos_token_id": 0,
                                           "splits": {"train": ["mmlu-0"], "development": ["mmlu-1"]}})
    pool = {"auxiliary_train": [row("mmlu", i) for i in range(0, 10)], "test": [row("mmlu", 5)]}
    monkeypatch.setattr(workload, "load_benchmark",
                        lambda c, splits=("test",): [r for s in splits for r in pool[s]])
    monkeypatch.setattr(workload, "load_tokenizer", lambda m: StubTokenizer())
    out = workload.extend({"base": str(base), "model": "x", "output": str(tmp_path / "out"),
                           "train_total": 6,
                           "sources": {"mmlu": ["auxiliary_train"]}, "exclude_test": ["mmlu"]})
    w = workload.Workload(out)
    train = w.ids("train")
    assert len(train) == 6 and train[0] == "mmlu-0" and w.ids("development") == ["mmlu-1"]
    assert "mmlu-5" not in train and "mmlu-1" not in train  # test overlap and dev excluded
    assert len(set(train)) == 6
