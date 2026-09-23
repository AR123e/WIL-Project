"""
Preliminary evaluation of the retrieval stage of the RAG pipeline.

Follows Walert's known/inferred question split. hit@1 / hit@3 check whether
a chunk that actually contains the answer shows up in the top-1 / top-3
retrieved chunks. "unanswered" is a simple confidence proxy: if the top
retrieval score falls below a threshold, we'd rather the system decline to
answer than risk an ungrounded response.
"""
import csv
import json
from pathlib import Path

from retrieve import ChessRetriever

EVAL_DIR = Path(__file__).parent.parent / "eval"
UNANSWERED_THRESHOLD = 5.0


def main():
    questions = json.loads((EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"))
    retriever = ChessRetriever()

    rows = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"], k=3)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        top1_score = retrieved[0]["score"] if retrieved else 0.0

        hit_at_1 = retrieved_ids[0] in q["expected_chunk_ids"] if retrieved_ids else False
        hit_at_3 = any(cid in q["expected_chunk_ids"] for cid in retrieved_ids)
        unanswered = top1_score < UNANSWERED_THRESHOLD

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "top1_chunk_id": retrieved_ids[0] if retrieved_ids else "",
            "top1_score": round(top1_score, 3),
            "expected_chunk_ids": ";".join(q["expected_chunk_ids"]),
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "unanswered": unanswered,
        })

    out_path = EVAL_DIR / "results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    def pct(pred):
        return 100 * sum(1 for r in rows if pred(r)) / n

    print(f"n = {n} questions")
    print(f"hit@1 overall: {pct(lambda r: r['hit_at_1']):.1f}%")
    print(f"hit@3 overall: {pct(lambda r: r['hit_at_3']):.1f}%")
    print(f"unanswered overall: {pct(lambda r: r['unanswered']):.1f}%")
    print()
    for t in ["known", "inferred"]:
        sub = [r for r in rows if r["type"] == t]
        hit1 = 100 * sum(1 for r in sub if r["hit_at_1"]) / len(sub)
        hit3 = 100 * sum(1 for r in sub if r["hit_at_3"]) / len(sub)
        print(f"[{t:8s}] n={len(sub)}  hit@1={hit1:.1f}%  hit@3={hit3:.1f}%")

    print(f"\nWrote per-question results to {out_path}")


if __name__ == "__main__":
    main()