"""
Retrieval component of the RAG pipeline. Uses BM25 over the chunked KB.
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class ChessRetriever:
    def __init__(self, chunks_path=CHUNKS_PATH):
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, k=3):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )[:k]
        results = []
        for i in ranked:
            results.append({**self.chunks[i], "score": float(scores[i])})
        return results