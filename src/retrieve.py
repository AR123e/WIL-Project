"""
Retrieval component of the RAG pipeline: BM25 over the chunked KB.

Improvements over the first version
- stop-word removal + light stemming so "moves"/"move", "pieces"/"piece" match
- the chunk's section title is indexed with its text (helps "opening principles" style queries)
- each result also carries `coverage` = share of query content words found in the chunk,
  a second confidence signal used to decline out-of-scope questions
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH

STOP = set("""a an and are as at be but by can could do does for from had has have how i if in into is it its
me my of on or our should so than that the their them then there these they this to us was we were what when
where which who whom why will with would you your tell about please explain give chess""".split())


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class ChessRetriever:
    def __init__(self, chunks_path=CHUNKS_PATH):
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_tokens = [tokenize(c["section"] + " " + c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.chunk_tokens)

    def retrieve(self, query, k=3):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in ranked:
            toks = set(self.chunk_tokens[i])
            coverage = (sum(1 for t in set(q) if t in toks) / len(set(q))) if q else 0.0
            results.append({**self.chunks[i], "score": float(scores[i]), "coverage": round(coverage, 3)})
        return results
