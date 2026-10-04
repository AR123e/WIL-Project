"""
Evaluation framework for the Chess Mentor RAG system (question types follow Walert: known / inferred, plus
overview and out_of_scope).

  python -m src.evaluate --stage retrieval --split dev          # fast, no LLM needed
  python -m src.evaluate --stage retrieval --split test --write-report   # final numbers -> eval/RESULTS.md
  python -m src.evaluate --compare --split dev                  # BM25 vs hybrid retriever, same gate
  python -m src.evaluate --stage full --baseline                # needs Ollama: answers, faithfulness, citations
  python -m src.evaluate --stage full --split test --limit 10   # quick smoke run on the first 10 questions
  python -m src.evaluate --export-review                        # sample answers for a HUMAN faithfulness check
  python -m src.evaluate --agreement                            # judge-vs-human agreement (Cohen's kappa)

Every run also appends a summary line to eval/history.jsonl, which the dashboard plots over time.

Dimensions
  Retrieval     hit@1, hit@3, MRR, nDCG@3  (a chunk is relevant if it contains an evidence phrase, so results
                survive re-chunking) with bootstrap 95% confidence intervals
  Effectiveness correct-refusal rate (out-of-scope) and false-refusal rate (in-scope)
  Correctness   keyword coverage of the answer vs gold keywords; RAG vs no-retrieval baseline
  Faithfulness  LLM judge: are all claims supported by the retrieved context?
  Attribution   citation present / valid (cited id was retrieved) / correct (cited id is an evidence chunk)
  Usability     Flesch reading ease, latency

Methodology: thresholds are tuned on the DEV split only (src/calibrate.py); the TEST split is for the final report.
"""
import argparse
import csv
import json
import math
import random
import re
from statistics import mean

from src import config
from src.config import EVAL_DIR, TOP_K

CITE = re.compile(r"\[([a-z]+_\d{3})\]")


# ---------------------------------------------------------------------------------------------------
# data + metric helpers (pure functions, unit-tested)
# ---------------------------------------------------------------------------------------------------
def load_questions(split="all"):
    qs = json.loads((EVAL_DIR / "test_questions.json").read_text(encoding="utf-8"))
    return qs if split == "all" else [q for q in qs if q.get("split") == split]


def evidence_ids(question, chunks):
    return {c["chunk_id"] for c in chunks if any(e in c["text"].lower() for e in question["evidence"])}


def ndcg_at_k(ranked_ids, relevant, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values, n=2000, seed=0):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


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


def retrieval_rows(retriever, questions, k=TOP_K, gate=None):
    """One row per question: ranking quality plus whether the confidence gate would decline it."""
    from src.pipeline import decide
    gate = gate or {}
    rows = []
    for q in questions:
        in_scope = q["type"] != "out_of_scope"
        retrieved = retriever.retrieve(q["question"], k=k)
        ev = evidence_ids(q, retriever.chunks)
        ids = [r["chunk_id"] for r in retrieved]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in ev), None)
        declined, reason = decide(q["question"], retrieved, **gate)
        rows.append({
            "id": q["id"], "type": q["type"], "split": q.get("split", ""), "question": q["question"],
            "top1_chunk": ids[0] if ids else "", "routed": (retrieved[0]["routed"] or "") if retrieved else "",
            "best_bm25": round(max((r["bm25"] for r in retrieved), default=0.0), 2),
            "best_char_sim": round(max((r["char_sim"] for r in retrieved), default=0.0), 2),
            "oov_ratio": retrieved[0]["oov_ratio"] if retrieved else 1.0,
            "evidence_chunks": ";".join(sorted(ev)),
            "hit_at_1": (rank == 1) if in_scope else None,
            "hit_at_3": (rank is not None) if in_scope else None,
            "rr": (1 / rank if rank else 0.0) if in_scope else None,
            "ndcg_at_3": ndcg_at_k(ids, ev, 3) if in_scope else None,
            "unanswered": declined, "decline_reason": reason if declined else "",
        })
    return rows


def summarize(rows):
    ins = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    out = {"n": len(rows), "n_in_scope": len(ins), "n_out_of_scope": len(oos)}
    if ins:
        out.update(hit1=pct([r["hit_at_1"] for r in ins]), hit3=pct([r["hit_at_3"] for r in ins]),
                   mrr=mean(r["rr"] for r in ins), ndcg3=mean(r["ndcg_at_3"] for r in ins),
                   hit3_ci=tuple(100 * x for x in bootstrap_ci([r["hit_at_3"] for r in ins])),
                   mrr_ci=bootstrap_ci([r["rr"] for r in ins]),
                   false_refusal=pct([r["unanswered"] for r in ins]))
    if oos:
        out["correct_refusal"] = pct([r["unanswered"] for r in oos])
    return out


def format_summary(name, rows):
    s = summarize(rows)
    lines = [f"{name}: n={s['n']} (in-scope {s['n_in_scope']}, out-of-scope {s['n_out_of_scope']})"]
    if "hit1" in s:
        lines.append(f"  retrieval   hit@1 {s['hit1']:5.1f}%   hit@3 {s['hit3']:5.1f}% (95% CI {s['hit3_ci'][0]:.0f}-{s['hit3_ci'][1]:.0f})"
                     f"   MRR {s['mrr']:.2f} (95% CI {s['mrr_ci'][0]:.2f}-{s['mrr_ci'][1]:.2f})   nDCG@3 {s['ndcg3']:.2f}")
        lines.append(f"  gate        false refusals {s['false_refusal']:5.1f}% of in-scope")
    if "correct_refusal" in s:
        lines.append(f"  gate        correct refusals {s['correct_refusal']:5.1f}% of out-of-scope")
    return "\n".join(lines)


def by_type_table(rows):
    lines = ["| type | n | hit@1 | hit@3 | MRR | nDCG@3 |", "|---|---|---|---|---|---|"]
    for t in ("known", "inferred", "overview"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            lines.append(f"| {t} | {len(sub)} | {pct([r['hit_at_1'] for r in sub]):.0f}% | {pct([r['hit_at_3'] for r in sub]):.0f}% | "
                         f"{mean(r['rr'] for r in sub):.2f} | {mean(r['ndcg_at_3'] for r in sub):.2f} |")
    return "\n".join(lines)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------
def run_full(questions, with_baseline, progress=None):
    from src.generate import generate_baseline, judge_faithfulness, get_llm
    from src.pipeline import ChessRAG
    if not get_llm().available(config.MODEL):
        raise SystemExit(f"Ollama model '{config.MODEL}' is not available. Run `ollama serve` and `ollama pull {config.MODEL}`.")
    rag = ChessRAG()
    rows = []
    for n, q in enumerate(questions, 1):
        if progress:
            progress(n, len(questions), q["question"])
        in_scope = q["type"] != "out_of_scope"
        res = rag.ask(q["question"])
        ev = evidence_ids(q, rag.retriever.chunks)
        ids = [c["chunk_id"] for c in res["chunks"]]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in ev), None)
        row = {"id": q["id"], "type": q["type"], "split": q.get("split", ""), "question": q["question"],
               "answer": res["answer"], "declined": res["declined"], "reason": res["reason"],
               "latency_s": round(res["latency"], 1),
               "hit_at_1": (rank == 1) if in_scope else None, "hit_at_3": (rank is not None) if in_scope else None,
               "rr": (1 / rank if rank else 0.0) if in_scope else None,
               "ndcg_at_3": ndcg_at_k(ids, ev, 3) if in_scope else None, "unanswered": res["declined"]}
        if not res["declined"]:
            cited = set(res["cited"])
            row.update(cite_present=bool(cited or res["invalid_citations"]), cite_valid=bool(cited) and not res["invalid_citations"],
                       cite_correct=bool(cited & ev) if in_scope else False,
                       faithful=judge_faithfulness(res["answer"], res["chunks"]),
                       keyword_score=keyword_score(res["answer"], q["keywords"]) if in_scope else None,
                       flesch=round(flesch(res["answer"]), 1))
        if with_baseline and in_scope:
            b = generate_baseline(q["question"])
            row.update(baseline_answer=b, baseline_keyword_score=keyword_score(b, q["keywords"]))
        rows.append(row)
    return rows


def summarize_full(rows, with_baseline=False):
    """Generation-stage metrics as a dict (used by the CLI, the dashboard and eval/history.jsonl)."""
    ins = [r for r in rows if r["type"] != "out_of_scope"]
    ans = [r for r in ins if not r["declined"]]
    out = {"answered": len(ans), "in_scope": len(ins), "latency_mean": mean(r["latency_s"] for r in rows) if rows else float("nan")}
    if ans:
        out.update(faithfulness=pct([r["faithful"] for r in ans]), citation_present=pct([r["cite_present"] for r in ans]),
                   citation_valid=pct([r["cite_valid"] for r in ans]), citation_correct=pct([r["cite_correct"] for r in ans]),
                   keyword_correctness=pct([r["keyword_score"] for r in ans]), flesch=mean(r["flesch"] for r in ans))
    if with_baseline:
        out.update(baseline_keyword=pct([r.get("baseline_keyword_score") for r in ins]),
                   rag_keyword=pct([(r["keyword_score"] if not r["declined"] else 0) for r in ins]))
    return out


def print_full(rows, with_baseline):
    s = summarize_full(rows, with_baseline)
    print("\n--- Generation (in-scope, answered) ---")
    print(f"answered: {s['answered']}/{s['in_scope']}")
    if "faithfulness" in s:
        print(f"faithfulness (LLM judge):     {s['faithfulness']:5.1f}%")
        print(f"citation present:             {s['citation_present']:5.1f}%")
        print(f"citation valid:               {s['citation_valid']:5.1f}%")
        print(f"citation correct:             {s['citation_correct']:5.1f}%")
        print(f"keyword correctness:          {s['keyword_correctness']:5.1f}%")
        print(f"Flesch reading ease (mean):   {s['flesch']:5.1f}  (60+ = plain English)")
    print(f"latency mean:                 {s['latency_mean']:5.1f}s")
    if with_baseline:
        print(f"\nBaseline (no retrieval) keyword correctness: {s['baseline_keyword']:5.1f}%  vs RAG {s['rag_keyword']:5.1f}%")


HISTORY_PATH = EVAL_DIR / "history.jsonl"


def append_history(stage, split, rows, with_baseline=False, path=None):
    """Append one summary line per evaluation run so the dashboard can show trends across code changes."""
    import time
    entry = {"ts": time.time(), "stage": stage, "split": split, "retriever": config.RETRIEVER, "model": config.MODEL,
             "n": len(rows), **{k: v for k, v in summarize(rows).items() if isinstance(v, (int, float))}}
    if stage == "full":
        entry.update({k: v for k, v in summarize_full(rows, with_baseline).items() if isinstance(v, (int, float))})
    p = path or HISTORY_PATH
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_history(path=None):
    p = path or HISTORY_PATH
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_and_save(stage, split, with_baseline=False, limit=None, progress=None):
    """Shared by the CLI and the dashboard button. Returns (rows, csv_path)."""
    from src.retrieve import ChessRetriever
    questions = load_questions(split)[:limit] if limit else load_questions(split)
    if stage == "retrieval":
        rows = retrieval_rows(ChessRetriever(), questions)
        out = EVAL_DIR / f"results_retrieval_{split}.csv"
    else:
        rows = run_full(questions, with_baseline, progress)
        out = EVAL_DIR / f"results_full_{split}.csv"
    write_csv(out, rows)
    append_history(stage, split, rows, with_baseline)
    return rows, out


# ---- human check of the LLM judge ----------------------------------------------------------------------
REVIEW_PATH = EVAL_DIR / "manual_review.csv"


def export_review(results_csv, n=15, seed=0, out=REVIEW_PATH):
    """Sample answered rows for a human to label (fill the human_faithful column with 1 = supported, 0 = not)."""
    rows = [r for r in csv.DictReader(open(results_csv, encoding="utf-8")) if r.get("declined") == "False" and r.get("answer")]
    random.Random(seed).shuffle(rows)
    keep = rows[:n]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer", "judge_faithful", "human_faithful"])
        w.writeheader()
        for r in keep:
            w.writerow({"id": r["id"], "question": r["question"], "answer": r["answer"],
                        "judge_faithful": r.get("faithful", ""), "human_faithful": ""})
    return len(keep)


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(v) / n) * (b.count(v) / n) for v in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(review_csv=REVIEW_PATH):
    rows = [r for r in csv.DictReader(open(review_csv, encoding="utf-8")) if r["human_faithful"].strip() in ("0", "1")]
    judge = [str(r["judge_faithful"]).strip().lower() in ("true", "1") for r in rows]
    human = [r["human_faithful"].strip() == "1" for r in rows]
    return {"n": len(rows), "agreement": 100 * sum(j == h for j, h in zip(judge, human)) / len(rows) if rows else float("nan"),
            "kappa": cohen_kappa(judge, human)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--compare", action="store_true", help="compare BM25-only vs hybrid retrieval")
    ap.add_argument("--write-report", action="store_true", help="write eval/RESULTS.md (retrieval stage)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick smoke run)")
    ap.add_argument("--export-review", action="store_true", help="sample answers for a human faithfulness check")
    ap.add_argument("--agreement", action="store_true", help="agreement between the LLM judge and your manual labels")
    args = ap.parse_args()
    if args.export_review:
        src_csv = EVAL_DIR / f"results_full_{args.split}.csv"
        print(f"wrote {export_review(src_csv)} answers to {REVIEW_PATH} - fill the human_faithful column (1/0), then run --agreement")
        return
    if args.agreement:
        a = agreement()
        print(f"judge vs human: n={a['n']}  agreement {a['agreement']:.0f}%  Cohen's kappa {a['kappa']:.2f}")
        return
    questions = load_questions(args.split)[:args.limit] if args.limit else load_questions(args.split)
    from src.retrieve import ChessRetriever

    if args.compare:
        for mode in ("bm25", "hybrid"):
            print(format_summary(f"[{mode}] split={args.split}", retrieval_rows(ChessRetriever(mode=mode), questions)))
        return

    if args.stage == "retrieval":
        rows, out = run_and_save("retrieval", args.split, limit=args.limit)
        print(format_summary(f"split={args.split}  retriever={config.RETRIEVER}", rows))
        print("\n" + by_type_table(rows))
        misses = [r for r in rows if r["type"] != "out_of_scope" and not r["hit_at_3"]]
        wrong = [r for r in rows if (r["type"] == "out_of_scope") != bool(r["unanswered"])]
        print(f"\nretrieval misses (not in top-3): {[r['id'] for r in misses]}")
        print(f"gate mistakes: {[(r['id'], r['question'][:40]) for r in wrong]}")
        if args.write_report:
            s = summarize(rows)
            (EVAL_DIR / "RESULTS.md").write_text(
                f"# Retrieval results ({args.split} split, retriever={config.RETRIEVER})\n\n"
                f"Questions: {s['n']} ({s['n_in_scope']} in-scope, {s['n_out_of_scope']} out-of-scope). "
                f"Thresholds were tuned on the dev split only.\n\n"
                f"- hit@1 {s['hit1']:.1f}%, hit@3 {s['hit3']:.1f}% (95% CI {s['hit3_ci'][0]:.0f}-{s['hit3_ci'][1]:.0f}), "
                f"MRR {s['mrr']:.2f}, nDCG@3 {s['ndcg3']:.2f}\n"
                f"- false refusals {s['false_refusal']:.1f}% of in-scope; correct refusals {s.get('correct_refusal', float('nan')):.1f}% of out-of-scope\n\n"
                f"{by_type_table(rows)}\n", encoding="utf-8")
            print("wrote eval/RESULTS.md")
        print(f"Per-question results: {out}")
        return

    def show(i, n, q):
        print(f"[{i}/{n}] {q}", flush=True)
    rows, out = run_and_save("full", args.split, args.baseline, args.limit, progress=show)
    print(format_summary(f"split={args.split}", rows))
    print_full(rows, args.baseline)
    print(f"Per-question results: {out}")


if __name__ == "__main__":
    main()
