import time

from src import config
from src.generate import generate_answer, split_citations, DECLINE
from src.retrieve import ChessRetriever
from src.text import tokenize


def decide(question, chunks, min_score=None, min_char_sim=None, min_coverage=None, min_wcoverage=None, max_oov=None):
    """Confidence gate. Returns (decline: bool, reason: str).

    Whole-game questions that were routed to the overview never decline. Otherwise decline only when BOTH
    retrieval signals are weak (best BM25 score AND best character-n-gram similarity), or the query has no
    content words at all (e.g. "what is it?").
    """
    min_score = config.MIN_SCORE if min_score is None else min_score
    min_char_sim = config.MIN_CHAR_SIM if min_char_sim is None else min_char_sim
    min_coverage = config.MIN_COVERAGE if min_coverage is None else min_coverage
    min_wcoverage = config.MIN_WCOVERAGE if min_wcoverage is None else min_wcoverage
    max_oov = config.MAX_OOV if max_oov is None else max_oov
    if not chunks:
        return True, "no_chunks"
    if chunks[0].get("routed"):
        return False, "routed_overview"
    if not tokenize(question, query=True):
        return True, "no_content_words"
    if max(c["bm25"] for c in chunks) < min_score and max(c["char_sim"] for c in chunks) < min_char_sim:
        return True, "low_retrieval_confidence"
    if chunks[0]["oov_ratio"] > max_oov:
        return True, "unknown_terms"
    if chunks[0]["coverage"] < min_coverage or chunks[0]["wcoverage"] < min_wcoverage:
        return True, "low_coverage"
    return False, "ok"


class ChessRAG:
    def __init__(self, retriever=None):
        self.retriever = retriever or ChessRetriever()

    def ask(self, question, k=config.TOP_K):
        t0 = time.time()
        chunks = self.retriever.retrieve(question, k=k)
        confidence = max((c["bm25"] for c in chunks), default=0.0)
        decline, reason = decide(question, chunks)
        base = {"chunks": chunks, "confidence": confidence, "routed": chunks[0]["routed"] if chunks else None}
        if decline:
            return {**base, "answer": DECLINE, "declined": True, "reason": reason, "cited": [], "invalid_citations": [],
                    "uncited": False, "latency": time.time() - t0}
        answer = generate_answer(question, chunks)
        declined = answer.lower().startswith("i don't know")
        cited, invalid = split_citations(answer, chunks)
        uncited = (not declined) and not cited
        if config.STRICT_CITATIONS and not declined and (uncited or invalid):
            answer, declined, reason = DECLINE, True, "citation_check_failed"
        return {**base, "answer": answer, "declined": declined, "reason": "llm_declined" if declined and reason == "ok" else reason,
                "cited": cited, "invalid_citations": invalid, "uncited": uncited, "latency": time.time() - t0}
