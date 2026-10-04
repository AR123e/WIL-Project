"""
Choose the confidence-gate thresholds on the DEV split only, so the TEST split stays untouched.

  python -m src.calibrate

Prints, for each candidate threshold, how many in-scope questions would be wrongly refused and how many
out-of-scope questions are correctly refused. Pick the setting with 0 false refusals and the most correct
refusals that also sits in the MIDDLE of its feasible range (a wide margin generalises better).
"""
from src.evaluate import load_questions, retrieval_rows, summarize
from src.retrieve import ChessRetriever


def sweep(name, values, key, retriever, questions, base):
    print(f"\n== {name} ==")
    best = []
    for v in values:
        s = summarize(retrieval_rows(retriever, questions, gate={**base, key: v}))
        print(f"  {key}={v:<5}  false refusals {s['false_refusal']:5.1f}%   correct refusals {s['correct_refusal']:5.1f}%")
        if s["false_refusal"] == 0:
            best.append((s["correct_refusal"], v))
    return best


if __name__ == "__main__":
    dev = load_questions("dev")
    r = ChessRetriever()
    base = {"min_score": 0.0, "min_char_sim": 0.0, "max_oov": 1.0}     # all optional gates off
    sweep("MIN_SCORE (BM25 floor)", [1.0, 2.0, 3.0, 3.5, 4.0, 5.0], "min_score", r, dev, {**base, "min_char_sim": 99})
    sweep("MAX_OOV (share of query words unknown to the KB)", [0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0], "max_oov", r, dev, base)
