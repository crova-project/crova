from crova import workload


class StubTokenizer:
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
