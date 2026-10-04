import pytest

from src import config
from src.generate import DECLINE, set_llm, split_citations, build_prompt
from src.llm import FakeLLM
from src.pipeline import ChessRAG


@pytest.fixture()
def rag(retriever):
    return ChessRAG(retriever=retriever)


def test_what_is_chess_goes_through_the_llm_with_citations(rag):
    llm = FakeLLM("Chess is a two-player board game [overview_001]. The goal is checkmate [overview_002].")
    set_llm(llm)
    res = rag.ask("What is chess?")
    assert not res["declined"] and res["reason"] == "routed_overview"
    assert res["cited"] == ["overview_001", "overview_002"] and not res["invalid_citations"] and not res["uncited"]
    assert "[overview_001]" in llm.calls[0]                   # the context really reached the prompt


def test_out_of_scope_is_declined_without_calling_the_llm(rag):
    llm = FakeLLM("should never be used")
    set_llm(llm)
    res = rag.ask("What is the capital of Australia?")
    assert res["declined"] and res["answer"] == DECLINE and llm.calls == []


def test_model_refusal_is_reported(rag):
    set_llm(FakeLLM(DECLINE))
    res = rag.ask("How does the knight move?")
    assert res["declined"] and res["reason"] == "llm_declined"


def test_invented_citation_is_flagged(rag):
    set_llm(FakeLLM("The knight jumps [fide_999]."))
    res = rag.ask("How does the knight move?")
    assert res["invalid_citations"] == ["fide_999"] and not res["cited"]


def test_uncited_answer_is_flagged_and_strict_mode_withholds_it(rag, monkeypatch):
    set_llm(FakeLLM("The knight moves in an L-shape."))
    assert rag.ask("How does the knight move?")["uncited"] is True
    monkeypatch.setattr(config, "STRICT_CITATIONS", True)
    res = rag.ask("How does the knight move?")
    assert res["declined"] and res["reason"] == "citation_check_failed"


def test_split_citations_separates_real_from_invented(retriever):
    chunks = retriever.retrieve("How does the knight move?", k=3)
    real = chunks[0]["chunk_id"]
    cited, invalid = split_citations(f"a [{real}] b [zzz_001] c [{real}]", chunks)
    assert cited == [real] and invalid == ["zzz_001"]


def test_prompt_contains_rules_context_and_question(retriever):
    chunks = retriever.retrieve("What is a fork?", k=3)
    p = build_prompt("What is a fork?", chunks)
    assert "ONLY the context" in p and chunks[0]["chunk_id"] in p and "Question: What is a fork?" in p
