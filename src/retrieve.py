"""Retrieval: BM25 over the chked KB, using the shared tokenizer in src/text.py"""
import json 
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.config import BM25_B, BM25_K1, CHUNKS_PATH
from src.text import classify_query, tokenize

# beginner-overview sections that answer whole-game questions ("What is chess?"), best first
ROUTES={
    "intro": ["What is chess?", "How to play chess: the quick version"],
    "howto": ["How to play chess: the quick version", "what is chess?"],
}

class ChessRetriever:
    def __init__(self, chunks_path=CHUNKS_PATH):
        self.chunks=json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_tokens=[tokenize(c["section"] + " " + c["text"]) for c in self.chunks]
        self.bm25=BM25Okapi(self.chunk_tokens, k1=BM25_K1, b=BM25_B)

    def _routed(self, intent):
        out=[]
        for section in ROUTES.get(intent, []):
            i=next((i for i, c in enumerate(self.chunks) if c["doc_id"]=="overview" and c["section"]==section), None)
            if i is not None:
                out.append(i)
        return out

    def retrieve(self, query, k=3):
        q=tokenize(query, expand=True, query=True)
        scores=self.bm25.get_scores(q) if q else [0.0] * len(self.chunks)
        ranked=sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        pinned=self._routed(classify_query(query))
        order=pinned+[i for i in ranked if i not in pinned]
        top_score=max(scores) if len(scores) else 0.0
        results=[]
        for i in order[:k]:
            toks=set(self.chunk_tokens[i])
            coverage=(sum(1 for t in set(q) if t in toks)/len(set(q))) if q else 0.0
            score=float(scores[i])
            if i in pinned:
                score, coverage=max(score, top_score, 99.0), 1.0
            results.append({**self.chunks[i], "score":score, "coverage": round(coverage, 3), "routed": i in pinned})
        return results