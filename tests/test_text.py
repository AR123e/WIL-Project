from src.text import tokenize, classify_query


def test_chess_is_dropped_from_queries_but_not_documents():
    assert tokenize("What is chess?", query=True) == []
    assert "chess" in tokenize("Chess is a board game")


def test_plural_and_singular_share_a_stem():
    assert tokenize("rules")[0] == tokenize("rule")[0]
    assert tokenize("pieces")[0] == tokenize("piece")[0]
    assert tokenize("castling")[0] == tokenize("castles")[0] == tokenize("castle")[0]


def test_british_american_spelling_is_unified():
    assert tokenize("centre") == tokenize("center")
    assert tokenize("defense") == tokenize("defence")


def test_function_word_only_query_has_no_tokens():
    assert tokenize("What is it?", query=True) == []


def test_query_expansion_adds_synonyms():
    assert "law" in tokenize("rules", expand=True)
    assert "improv" in tokenize("get better", expand=True)


def test_intent_routing():
    for q in ["What is chess?", "what is chess", "Tell me about chess", "Explain chess to me", "Teach me chess",
              "What is the game of chess?"]:
        assert classify_query(q) == "intro", q
    for q in ["How do I play chess?", "What are the rules of chess?", "I'm new to chess, where do I start?"]:
        assert classify_query(q) == "howto", q
    for q in ["How does the knight move?", "What is the objective of chess?", "Who invented chess?", "What is a fork?"]:
        assert classify_query(q) is None, q
