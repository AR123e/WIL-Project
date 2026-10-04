"""
Plain-English view of an evaluation results file, so the dashboard can show a simple scorecard instead of a
wall of metrics. Pure functions (no Streamlit) so they can be unit-tested.

Mapping from the technical metric to the label a non-expert sees:
  hit@3                -> "Finds the right source"
  hit@1                -> "Puts the best source first"
  1 - false refusals   -> "Answers valid questions"
  correct refusals     -> "Declines off-topic questions"
  faithfulness         -> "Answers backed by sources"      (full runs only; AI-judged)
  citation correct     -> "Cites a correct source"         (full runs only)
"""
import pandas as pd

TYPE_NAMES = {"known": "Direct questions", "inferred": "Rephrased questions", "overview": "Whole-game questions"}


def _num(series):
    """Booleans that went through CSV (True/False/NaN, possibly as text) -> 1.0 / 0.0 / NaN."""
    def one(v):
        if v is True or v == "True" or v == 1:
            return 1.0
        if v is False or v == "False" or v == 0:
            return 0.0
        return float("nan")
    return pd.to_numeric(series.map(one), errors="coerce")


def _pct(series):
    s = _num(series).dropna()
    return 100 * s.mean() if len(s) else None


def band(value):
    """Traffic light for a 0-100 score."""
    if value is None:
        return "⚪"
    return "🟢" if value >= 90 else "🟡" if value >= 75 else "🔴"


def scorecard(df):
    ins, oos = df[df["type"] != "out_of_scope"], df[df["type"] == "out_of_scope"]
    sc = {"n": len(df), "n_valid": len(ins), "n_offtopic": len(oos),
          "found3": _pct(ins["hit_at_3"]), "found1": _pct(ins["hit_at_1"]),
          "answers_valid": None, "declines_offtopic": _pct(oos["unanswered"]) if len(oos) else None,
          "backed": None, "cites_correct": None, "rag_keyword": None, "base_keyword": None}
    refused = _pct(ins["unanswered"])
    sc["answers_valid"] = None if refused is None else 100 - refused
    if "faithful" in df.columns:
        answered = ins[~_num(ins["declined"]).fillna(0).astype(bool)]
        sc["backed"] = _pct(answered["faithful"])
        sc["cites_correct"] = _pct(answered["cite_correct"])
        if "baseline_keyword_score" in df.columns:
            kw = ins.apply(lambda r: 0.0 if r["declined"] in (True, "True") else r["keyword_score"], axis=1)
            sc["rag_keyword"] = 100 * pd.to_numeric(kw, errors="coerce").mean()
            sc["base_keyword"] = 100 * pd.to_numeric(ins["baseline_keyword_score"], errors="coerce").mean()
    return sc


def tiles(sc):
    """Headline numbers: (label, value 0-100 or None, one-sentence explanation)."""
    out = [
        ("Finds the right source", sc["found3"],
         "For valid chess questions: how often the correct passage was among the 3 the system looked up."),
        ("Puts the best source first", sc["found1"],
         "How often the correct passage was the very first one found."),
        ("Answers valid questions", sc["answers_valid"],
         "Valid questions it did NOT wrongly refuse. 100% means it never says 'I don't know' to a question it should answer."),
        ("Declines off-topic questions", sc["declines_offtopic"],
         "Questions the knowledge base can't answer (clock rules, capital cities...) that it correctly declined instead of guessing."),
    ]
    if sc["backed"] is not None:
        out.append(("Answers backed by sources", sc["backed"],
                    "Share of answers whose claims are supported by the retrieved passages. Judged automatically by the AI model, "
                    "so check a sample by hand before relying on it."))
        out.append(("Cites a correct source", sc["cites_correct"],
                    "Answers that point to a passage which really contains the answer."))
    return out


def verdict(sc, filename=""):
    def f(v):
        return "n/a" if v is None else f"{v:.0f}%"
    text = (f"On {sc['n']} test questions ({sc['n_valid']} valid, {sc['n_offtopic']} off-topic), the mentor found the right "
            f"source for {f(sc['found3'])}, wrongly refused {f(None if sc['answers_valid'] is None else 100 - sc['answers_valid'])} "
            f"of valid questions, and correctly declined {f(sc['declines_offtopic'])} of off-topic ones.")
    notes = []
    if sc["n"] < 30:
        notes.append(f"Only {sc['n']} questions were checked, so treat this as a rough guide.")
    if "dev" in filename:
        notes.append("This file is from the dev split, which was used to tune the system. Use the test split for honest numbers.")
    return text, notes


def by_type(df):
    ins = df[df["type"] != "out_of_scope"]
    rows = {TYPE_NAMES.get(t, t): _pct(g["hit_at_3"]) for t, g in ins.groupby("type")}
    return pd.Series({k: v for k, v in rows.items() if v is not None})


def struggles(df, limit=10):
    """The questions the system got wrong, with a plain-language reason."""
    rows = []
    for _, r in df.iterrows():
        in_scope = r["type"] != "out_of_scope"
        refused = r["unanswered"] in (True, "True")
        if in_scope and r["hit_at_3"] in (False, "False"):
            rows.append((r["question"], TYPE_NAMES.get(r["type"], r["type"]), "Right source not found in the top 3"))
        elif in_scope and refused:
            rows.append((r["question"], TYPE_NAMES.get(r["type"], r["type"]), "Wrongly said 'I don't know'"))
        elif not in_scope and not refused:
            rows.append((r["question"], "Off-topic", "Should have declined but tried to answer"))
        elif in_scope and "faithful" in r and r["faithful"] in (False, "False"):
            rows.append((r["question"], TYPE_NAMES.get(r["type"], r["type"]), "Answer not fully backed by its sources"))
    return pd.DataFrame(rows[:limit], columns=["Question", "Type", "What went wrong"])


def pick_default_file(files):
    """Prefer honest (test-split) results, full runs before retrieval-only, otherwise the newest file."""
    names = {f.name: f for f in files}
    for preferred in ("results_full_test.csv", "results_retrieval_test.csv"):
        if preferred in names:
            return names[preferred]
    return max(files, key=lambda f: f.stat().st_mtime)
