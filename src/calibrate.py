"""Shows BM25 score / coverage of the top chunk per question type to help pick MIN_SCORE and MIN_COVERAGE."""
import json
from statistics import mean

from src.config import EVAL_DIR
from src.retrieve import ChessRetriever

qs = json.loads((EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"))
r = ChessRetriever()
by = {}
for q in qs:
    top = r.retrieve(q["question"], k=1)[0]
    by.setdefault(q["type"], []).append((top["score"], top["coverage"], q["question"]))
for t, rows in by.items():
    print(f"\n== {t}: mean score {mean(x[0] for x in rows):.2f}, mean coverage {mean(x[1] for x in rows):.2f}")
    for s, c, q in sorted(rows):
        print(f"  {s:6.2f}  cov={c:.2f}  {q}")
