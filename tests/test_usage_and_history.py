import csv
import json

import pytest

from src import evaluate
from src.generate import set_llm
from src.llm import FakeLLM
from src.pipeline import ChessRAG
from src.usage_log import log_question, log_feedback, read_events, usage_summary


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    p = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CHESS_LOG_PATH", str(p))
    return p


def test_usage_log_roundtrip_and_summary(log_file, retriever):
    rag = ChessRAG(retriever=retriever)
    set_llm(FakeLLM("Chess is a board game [overview_001]."))
    a = rag.ask("What is chess?")
    b = rag.ask("What is the capital of Australia?")            # declined by the gate
    ida, idb = log_question("What is chess?", a), log_question("What is the capital of Australia?", b)
    log_feedback(ida, "up")
    log_feedback(idb, "down")
    log_feedback(idb, "up")                                      # a later vote replaces the earlier one
    s = usage_summary(read_events())
    assert s["n_questions"] == 2 and s["decline_rate"] == 50
    assert s["thumbs_up"] == 2 and s["thumbs_down"] == 0
    assert s["reasons"]["low_retrieval_confidence"] == 1 and s["routes"]["intro"] == 1


def test_corrupt_log_line_does_not_break_the_dashboard(log_file):
    log_file.write_text('{"type": "ask", "id": "a", "declined": false, "uncited": false, "latency": 1.0, '
                        '"reason": null, "routed": null}\n{"type": "as', encoding="utf-8")
    assert usage_summary(read_events())["n_questions"] == 1


def test_empty_log_is_fine(log_file):
    assert usage_summary(read_events())["n_questions"] == 0


def test_full_stage_runs_end_to_end_with_a_fake_llm(tmp_path, monkeypatch):
    """The generation-stage evaluation (citations, faithfulness judge, baseline) never needs Ollama in tests."""
    monkeypatch.setattr(evaluate, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "HISTORY_PATH", tmp_path / "history.jsonl")
    (tmp_path / "test_questions.json").write_text(
        (evaluate.config.EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"), encoding="utf-8")

    def reply(prompt):
        if "strict fact-checker" in prompt:
            return "SUPPORTED"
        if "Context:" not in prompt:
            return "baseline answer with an L-shape"
        cid = prompt.split("Context:")[1].split("]")[0].strip().lstrip("[")
        return f"Answer from the text [{cid}]."
    set_llm(FakeLLM(reply))
    progress = []
    rows, out = evaluate.run_and_save("full", "dev", with_baseline=True, limit=8, progress=lambda i, n, q: progress.append(i))
    assert progress == list(range(1, 9)) and out.exists()
    s = evaluate.summarize_full(rows, with_baseline=True)
    assert s["faithfulness"] == 100 and s["citation_valid"] == 100
    hist = evaluate.read_history(tmp_path / "history.jsonl")
    assert len(hist) == 1 and hist[0]["stage"] == "full" and "faithfulness" in hist[0]


def test_cohen_kappa_and_agreement(tmp_path):
    assert evaluate.cohen_kappa([True, False, True], [True, False, True]) == 1.0
    p = tmp_path / "m.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer", "judge_faithful", "human_faithful"])
        w.writeheader()
        for i, (j, h) in enumerate([("True", "1"), ("True", "1"), ("True", "0"), ("False", "0"), ("True", "1"), ("False", "1")]):
            w.writerow({"id": i, "question": "q", "answer": "a", "judge_faithful": j, "human_faithful": h})
    a = evaluate.agreement(p)
    assert a["n"] == 6 and round(a["agreement"]) == 67 and round(a["kappa"], 2) == 0.25
