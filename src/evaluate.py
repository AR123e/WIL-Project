"""
Evaluation framework for the Chess Mentor RAG system.

Run:
  python -m src.evaluate --stage retrieval      # fast, no LLM needed
  python -m src.evaluate --stage full           # needs Ollama running (answers + faithfulness judge)
  python -m src.evaluate --stage full --baseline  # also runs the no-retrieval LLM baseline

Dimensions measured (question types follow Walert: known / inferred, plus out_of_scope)
  Retrieval     hit@1, hit@3, MRR  (a chunk is relevant if it contains an evidence phrase)
  Effectiveness % unanswered, correct-refusal rate (out_of_scope), false-refusal rate (in-scope)
  Correctness   keyword coverage of the answer vs gold keywords (in-scope, answered)
  Faithfulness  LLM judge: are all claims supported by the retrieved context?
  Attribution   citation present / valid (cited id was retrieved) / correct (cited id is an evidence chunk)
  Usability     Flesch reading ease (beginner-friendliness), latency
"""
import argparse
import csv
import json
import re
from statistics import mean

from src.config import EVAL_DIR, TOP_K
from src.retrieve import ChessRetriever

CITE = re.compile(r"\[([a-z]+_\d{3})\]")


def evidence_ids(question, chunks):
    return {c["chunk_id"] for c in chunks if any(e in c["text"].lower() for e in question["evidence"])}


def flesch(text):
    words = re.findall(r"[A-Za-z']+", text)
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    if not words:
        return 0.0
    syll = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (syll / len(words))


def keyword_score(answer, keywords):
    if not keywords:
        return None
    a = answer.lower()
    return sum(1 for k in keywords if any(alt in a for alt in k.split("|"))) / len(keywords)


def pct(vals):
    vals = [v for v in vals if v is not None]
    return 100 * mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    questions = json.loads((EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"))
    rag = None
    if args.stage == "full":
        from src.pipeline import ChessRAG
        from src.generate import generate_baseline, judge_faithfulness
        rag = ChessRAG()
        retriever = rag.retriever
    else:
        retriever = ChessRetriever()

    from src.config import MIN_SCORE, MIN_COVERAGE
    rows = []
    for q in questions:
        in_scope = q["type"] != "out_of_scope"
        retrieved = retriever.retrieve(q["question"], k=TOP_K)
        ev = evidence_ids(q, retriever.chunks)
        ids = [r["chunk_id"] for r in retrieved]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in ev), None)
        row = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "top1_score": round(retrieved[0]["score"], 2), "top1_chunk": ids[0],
            "evidence_chunks": ";".join(sorted(ev)),
            "hit_at_1": (rank == 1) if in_scope else None,
            "hit_at_3": (rank is not None) if in_scope else None,
            "rr": (1 / rank if rank else 0.0) if in_scope else None,
            "unanswered": retrieved[0]["score"] < MIN_SCORE or retrieved[0]["coverage"] < MIN_COVERAGE,
        }
        if args.stage == "full":
            res = rag.ask(q["question"])
            ans = res["answer"]
            declined = res["declined"]
            row.update({"answer": ans, "declined": declined, "latency_s": round(res["latency"], 1)})
            row["unanswered"] = declined
            if not declined:
                cited = set(CITE.findall(ans))
                row["cite_present"] = bool(cited)
                row["cite_valid"] = bool(cited) and cited <= set(ids)
                row["cite_correct"] = bool(cited & ev) if in_scope else False
                row["faithful"] = judge_faithfulness(ans, res["chunks"])
                row["keyword_score"] = keyword_score(ans, q["keywords"]) if in_scope else None
                row["flesch"] = round(flesch(ans), 1)
            if args.baseline and in_scope:
                b = generate_baseline(q["question"])
                row["baseline_answer"] = b
                row["baseline_keyword_score"] = keyword_score(b, q["keywords"])
        rows.append(row)

    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    out = EVAL_DIR / f"results_{args.stage}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ins = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    print(f"\nn = {len(rows)}  (known {sum(r['type']=='known' for r in rows)}, "
          f"inferred {sum(r['type']=='inferred' for r in rows)}, out_of_scope {len(oos)})")
    print("\n--- Retrieval (in-scope) ---")
    for label, sub in [("all", ins)] + [(t, [r for r in ins if r["type"] == t]) for t in ("known", "inferred")]:
        print(f"[{label:8s}] n={len(sub):2d} hit@1={pct([r['hit_at_1'] for r in sub]):5.1f}%  "
              f"hit@3={pct([r['hit_at_3'] for r in sub]):5.1f}%  MRR={mean(r['rr'] for r in sub):.2f}")
    print("\n--- Effectiveness ---")
    print(f"unanswered (all questions):        {pct([r['unanswered'] for r in rows]):5.1f}%")
    print(f"false refusals (in-scope):          {pct([r['unanswered'] for r in ins]):5.1f}%")
    print(f"correct refusals (out-of-scope):    {pct([r['unanswered'] for r in oos]):5.1f}%")

    if args.stage == "full":
        ans = [r for r in ins if not r["declined"]]
        print("\n--- Generation (in-scope, answered) ---")
        print(f"answered: {len(ans)}/{len(ins)}")
        print(f"faithfulness (LLM judge):     {pct([r['faithful'] for r in ans]):5.1f}%")
        print(f"citation present:             {pct([r['cite_present'] for r in ans]):5.1f}%")
        print(f"citation valid:               {pct([r['cite_valid'] for r in ans]):5.1f}%")
        print(f"citation correct:             {pct([r['cite_correct'] for r in ans]):5.1f}%")
        print(f"keyword correctness:          {pct([r['keyword_score'] for r in ans]):5.1f}%")
        print(f"Flesch reading ease (mean):   {mean(r['flesch'] for r in ans):5.1f}  (60+ = plain English)")
        print(f"latency mean:                 {mean(r['latency_s'] for r in rows):5.1f}s")
        if args.baseline:
            print(f"\nBaseline (no retrieval) keyword correctness: "
                  f"{pct([r.get('baseline_keyword_score') for r in ins]):5.1f}%  vs RAG {pct([r['keyword_score'] if not r['declined'] else 0 for r in ins]):5.1f}%")
    print(f"\nPer-question results: {out}")


if __name__ == "__main__":
    main()
