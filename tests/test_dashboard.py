import pandas as pd

from src import dashboard


def retrieval_df():
    return pd.DataFrame([
        {"id": "a", "type": "known", "question": "q1", "hit_at_1": True, "hit_at_3": True, "unanswered": False},
        {"id": "b", "type": "known", "question": "q2", "hit_at_1": False, "hit_at_3": True, "unanswered": False},
        {"id": "c", "type": "inferred", "question": "q3", "hit_at_1": False, "hit_at_3": False, "unanswered": False},
        {"id": "d", "type": "inferred", "question": "q4", "hit_at_1": True, "hit_at_3": True, "unanswered": True},   # wrongly refused
        {"id": "e", "type": "out_of_scope", "question": "q5", "hit_at_1": None, "hit_at_3": None, "unanswered": True},
        {"id": "f", "type": "out_of_scope", "question": "q6", "hit_at_1": None, "hit_at_3": None, "unanswered": False},  # should refuse
    ])


def test_scorecard_translates_metrics_to_plain_numbers():
    sc = dashboard.scorecard(retrieval_df())
    assert (sc["n"], sc["n_valid"], sc["n_offtopic"]) == (6, 4, 2)
    assert sc["found3"] == 75 and sc["found1"] == 50
    assert sc["answers_valid"] == 75           # 1 of 4 valid questions wrongly refused
    assert sc["declines_offtopic"] == 50
    assert sc["backed"] is None                # retrieval-only file has no generation metrics


def test_tiles_add_generation_rows_only_for_full_runs():
    sc = dashboard.scorecard(retrieval_df())
    assert [t[0] for t in dashboard.tiles(sc)] == ["Finds the right source", "Puts the best source first",
                                                   "Answers valid questions", "Declines off-topic questions"]
    sc["backed"], sc["cites_correct"] = 80.0, 90.0
    assert len(dashboard.tiles(sc)) == 6


def test_full_results_with_csv_style_values():
    df = retrieval_df()
    df["declined"] = df["unanswered"]
    df["faithful"] = ["True", "False", "True", None, None, "True"]
    df["cite_correct"] = ["True", "True", "False", None, None, "False"]
    df["keyword_score"] = [1.0, 0.5, 0.0, 0.0, None, None]
    df["baseline_keyword_score"] = [0.5, 0.5, 0.0, 0.0, None, None]
    sc = dashboard.scorecard(df)
    assert round(sc["backed"]) == 67 and sc["cites_correct"] is not None
    assert sc["rag_keyword"] is not None and sc["base_keyword"] is not None


def test_struggles_lists_each_kind_of_mistake_in_plain_language():
    w = dashboard.struggles(retrieval_df())
    reasons = dict(zip(w["Question"], w["What went wrong"]))
    assert reasons["q3"] == "Right source not found in the top 3"
    assert reasons["q4"] == "Wrongly said 'I don't know'"
    assert reasons["q6"] == "Should have declined but tried to answer"
    assert "q1" not in reasons and "q5" not in reasons


def test_verdict_warns_on_small_or_dev_results():
    sc = dashboard.scorecard(retrieval_df())
    text, notes = dashboard.verdict(sc, "results_retrieval_dev.csv")
    assert "6 test questions" in text
    assert any("Only 6" in n for n in notes) and any("dev split" in n for n in notes)


def test_traffic_light_and_by_type():
    assert dashboard.band(95) == "🟢" and dashboard.band(80) == "🟡" and dashboard.band(50) == "🔴" and dashboard.band(None) == "⚪"
    bt = dashboard.by_type(retrieval_df())
    assert bt["Direct questions"] == 100 and bt["Rephrased questions"] == 50


def test_pick_default_prefers_test_results(tmp_path):
    a, b, c = tmp_path / "results_retrieval_dev.csv", tmp_path / "results_retrieval_test.csv", tmp_path / "results_full_dev.csv"
    for p in (a, b, c):
        p.write_text("x")
    assert dashboard.pick_default_file([a, b, c]).name == "results_retrieval_test.csv"
    assert dashboard.pick_default_file([a, c]).name in ("results_retrieval_dev.csv", "results_full_dev.csv")
