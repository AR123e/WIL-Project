"""
Retrieval component: BM25 (lexical) + character n-gram TF-IDF (spelling/morphology tolerant), fused with
Reciprocal Rank Fusion. Whole-game questions ("What is chess?") are routed to the beginner overview.

Each result carries the signals the confidence gate needs:
  bm25      raw BM25 score (keyword evidence)
  char_sim  cosine similarity of character n-grams (robust to word forms and typos)
  coverage  share of query content words found in the chunk
  wcoverage the same, weighted by term rarity (a never-seen query word counts as very informative)
  oov_ratio share of query content words that appear nowhere in the knowledge base
  fused     reciprocal-rank-fusion score used for ordering
"""
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CHUNKS_PATH, RETRIEVER, RRF_K
from src.text import tokenize, classify_query


class ChessRetriever:
    def __init__(self, chunks_path=CHUNKS_PATH, mode=RETRIEVER):
        if mode not in ("bm25", "hybrid"):
            raise ValueError("mode must be 'bm25' or 'hybrid'")
        self.mode = mode
        self.chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_tokens = [tokenize(c["section"] + " " + c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.chunk_tokens)
        self.vocab = {t for toks in self.chunk_tokens for t in toks}
        self.max_idf = float(np.log(len(self.chunks)))
        # character n-grams over the stemmed text, so "castling"/"castles"/"castle" and small typos still overlap
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_matrix = self.char_vec.fit_transform([" ".join(t) for t in self.chunk_tokens])

    # ---- scoring -----------------------------------------------------------------------------
    def _scores(self, query_tokens):
        bm = np.asarray(self.bm25.get_scores(query_tokens), dtype=float) if query_tokens else np.zeros(len(self.chunks))
        if query_tokens:
            q = self.char_vec.transform([" ".join(query_tokens)])
            cs = (self.char_matrix @ q.T).toarray().ravel()
        else:
            cs = np.zeros(len(self.chunks))
        return bm, cs

    def _weight(self, token):
        """IDF weight of a query term; a term the whole KB has never seen counts as maximally informative."""
        if token not in self.vocab:
            return self.max_idf
        return max(self.bm25.idf.get(token, 0.0), 0.1)

    @staticmethod
    def _rank_positions(scores):
        order = np.argsort(-scores, kind="stable")
        pos = np.empty(len(scores), dtype=int)
        pos[order] = np.arange(1, len(scores) + 1)
        return pos

    def _fuse(self, bm, cs):
        if self.mode == "bm25":
            return bm.copy()
        fused = np.zeros(len(bm))
        for s in (bm, cs):
            pos = self._rank_positions(s)
            fused += np.where(s > 0, 1.0 / (RRF_K + pos), 0.0)
        return fused

    # ---- public API --------------------------------------------------------------------------
    def retrieve(self, query, k=3):
        q_tokens = tokenize(query, expand=True, query=True)
        bm, cs = self._scores(q_tokens)
        fused = self._fuse(bm, cs)
        routed = classify_query(query)

        if routed == "intro":      # whole-game definition: the "What is chess?" section, in document order
            idx = [i for i, c in enumerate(self.chunks) if "intro" in c.get("tags", [])][:k]
        elif routed == "howto":    # how to start / basic rules: the quick-start section, then best-matching overview chunks
            first = [i for i, c in enumerate(self.chunks) if "howto" in c.get("tags", [])]
            rest = sorted((i for i, c in enumerate(self.chunks) if "overview" in c.get("tags", []) and i not in first),
                          key=lambda i: (-fused[i], i))
            idx = (first + rest)[:k]
        else:
            idx = sorted(range(len(self.chunks)), key=lambda i: (-fused[i], i))[:k]

        q_set = set(q_tokens)
        total_w = sum(self._weight(t) for t in q_set)
        oov = (sum(1 for t in q_set if t not in self.vocab) / len(q_set)) if q_set else 0.0
        results = []
        for i in idx:
            toks = set(self.chunk_tokens[i])
            coverage = (len(q_set & toks) / len(q_set)) if q_set else 0.0
            wcov = (sum(self._weight(t) for t in q_set & toks) / total_w) if total_w else 0.0
            results.append({**self.chunks[i], "score": float(bm[i]), "bm25": float(bm[i]),
                            "char_sim": float(cs[i]), "coverage": round(coverage, 3),
                            "wcoverage": round(wcov, 3), "oov_ratio": round(oov, 3),
                            "fused": float(fused[i]), "routed": routed})
        return results
