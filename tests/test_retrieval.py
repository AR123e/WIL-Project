from src.pipeline import decide
from src.evaluate import evidence_ids


def top_ids(r, q, k=3):
    return [c["chunk_id"] for c in r.retrieve(q, k=k)]


def test_what_is_chess_is_answered_not_declined(retriever):
    """Regression: 'chess', 'what' and 'is' used to be stop-words, so this became an empty query -> 'I don't know'."""
    chunks = retriever.retrieve("What is chess?", k=3)
    assert chunks[0]["routed"] == "intro"
    assert "two-player strategy board game" in chunks[0]["text"]
    assert decide("What is chess?", chunks) == (False, "routed_overview")


def test_howto_question_starts_with_the_quick_version(retriever):
    chunks = retriever.retrieve("How do I play chess?", k=3)
    assert "howto" in chunks[0]["tags"]
    assert decide("How do I play chess?", chunks)[0] is False


def test_plural_query_finds_rule_text(retriever):
    ids = top_ids(retriever, "How do rooks move?")
    assert any("rook" in c["text"].lower() for c in retriever.retrieve("How do rooks move?", k=3))
    assert ids


def test_misspelled_query_still_finds_castling(retriever):
    texts = [c["text"].lower() for c in retriever.retrieve("How does castelling work?", k=3)]
    assert any("castl" in t for t in texts)


def test_british_spelling_matches_american_query(retriever):
    texts = " ".join(c["text"].lower() for c in retriever.retrieve("Why control the center?", k=3))
    assert "centre" in texts


def test_glossary_terms_are_retrievable(retriever):
    for q, key in [("What is a fork?", "fork:"), ("What does zugzwang mean?", "zugzwang:"), ("What is a pin?", "pin:")]:
        assert any(c["text"].lower().startswith(key) for c in retriever.retrieve(q, k=3)), q


def test_off_topic_questions_are_declined(retriever):
    for q in ["What is the capital of Australia?", "Write me a poem about the sea", "Who invented chess?", "What is it?"]:
        chunks = retriever.retrieve(q, k=3)
        assert decide(q, chunks)[0] is True, q


def test_knight_question_finds_the_rule(retriever):
    assert any("l-shape" in c["text"].lower() for c in retriever.retrieve("How does the knight move?", k=3))
