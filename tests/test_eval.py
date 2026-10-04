import math

from src.evaluate import (load_questions, evidence_ids, ndcg_at_k, bootstrap_ci, summarize, keyword_score)


def test_every_in_scope_question_has_a_relevant_chunk(chunks_path):
    import json
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    for q in load_questions():
        if q["type"] != "out_of_scope":
            assert evidence_ids(q, chunks), f"{q['id']} has no chunk containing its evidence phrase"


def test_dev_test_split_is_complete_and_disjoint():
    dev, test = load_questions("dev"), load_questions("test")
    assert dev and test
    assert not {q["id"] for q in dev} & {q["id"] for q in test}
    assert len(dev) + len(test) == len(load_questions("all"))
    for split in (dev, test):
        assert {q["type"] for q in split} == {"known", "inferred", "overview", "out_of_scope"}


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["x", "y", "z"], {"a"}, 3) == 0.0
    assert math.isclose(ndcg_at_k(["x", "a", "z"], {"a"}, 3), 1 / math.log2(3))


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = [1, 1, 0, 1, 0, 1, 1, 1]
    lo, hi = bootstrap_ci(vals)
    assert (lo, hi) == bootstrap_ci(vals) and lo <= sum(vals) / len(vals) <= hi


def test_keyword_score_supports_alternatives():
    assert keyword_score("It moves in an L-shape", ["l-shape|l shape"]) == 1.0
    assert keyword_score("No idea", ["l-shape"]) == 0.0


def test_summarize_counts_refusals():
    rows = [
        {"type": "known", "hit_at_1": True, "hit_at_3": True, "rr": 1.0, "ndcg_at_3": 1.0, "unanswered": False},
        {"type": "out_of_scope", "hit_at_1": None, "hit_at_3": None, "rr": None, "ndcg_at_3": None, "unanswered": True},
    ]
    s = summarize(rows)
    assert s["false_refusal"] == 0 and s["correct_refusal"] == 100 and s["hit3"] == 100
