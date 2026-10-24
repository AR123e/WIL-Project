"""Offline regression tests. Run: python -m  pytest -q They lock in the fix for basic beginner questions that used to be declined."""
import pytest
from src.pipeline import ChessRAG
from src.text import classify_query, tokenize

@pytest.fixture(scope="module")
def rag():
    return ChessRAG()

# question, a phrase that must appear in one of the the top-3 chunks; "|" separates accepted alternatives

BASIC=[
    ("What is chess?", "two-player strategy board game"),
    ("How do I play chess?", "white move first"),
    ("What are the rules of chess?", "white move first"),
    ("How does a pawn move?", "a pawn moves forward one square"),
    ("How does the king move?", "the king moves one square in any direction"),
    ("How does a rook move?","rook may move to any square along the file or the rank|a rook moves any number of squares"),
    ("What is a checkmate?","checkmate"),
    ("what is a fork?", "fork:"),
    ("What is a pin?","pin:"),
    ("How do I get better at chess?", "to improve at chess"),
    ("How many pieces does each player have?", "16 pieces"),
]

@pytest.mark.parametrize("question,phrase", BASIC)
def test_basic_questions_are_answered(rag,question,phrase):
    chunks=rag.retriever.retrieve(question, k=3)
    assert not rag.should_decline(chunks), f"declined: {question}"
    assert any(alt in c["text"].lower() for c in chunks for alt in phrase.split("|")), f"no relevant chunk for: {question}"


@pytest.mark.parametrize("question", ["What is the capital of France?", "How is an Elo rating calculated"])
def test_off_topic_questions_are_decline(rag, question):
    assert rag.should_decline(rag.retriever.retrieve(question, k=3))

def test_all_raw_documents_are_indexed(rag):
    assert {c["doc_id"] for c in rag.retriever.chunks} >= {"fide","wikibooks","overview","strategy","tactics"}

def test_tokenizer_keeps_stems_consistent():
    assert tokenize("moves")==tokenize("move") and tokenize("pieces")==tokenize("piece")

def test_int_routing():
    assert classify_query("What is chess?")=="intro"
    assert classify_query("How do I play chess?")=="howto"
    assert classify_query("How does the knight move?") is None
  