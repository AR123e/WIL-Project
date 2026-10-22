import time

from src.config import TOP_K, MIN_SCORE, MIN_COVERAGE
from src.generate import generate_answer, DECLINE
from src.retrieve import ChessRetriever


class ChessRAG:
    def __init__(self):
        self.retriever = ChessRetriever()

    def should_decline(self, chunks):
        return (not chunks) or chunks[0]["score"] < MIN_SCORE or chunks[0]["coverage"] < MIN_COVERAGE

    def ask(self, question, k=TOP_K):
        t0 = time.time()
        chunks = self.retriever.retrieve(question, k=k)
        confidence = chunks[0]["score"] if chunks else 0.0
        if self.should_decline(chunks):
            return {"answer": DECLINE, "declined": True, "chunks": chunks,
                    "confidence": confidence, "latency": time.time() - t0}
        answer = generate_answer(question, chunks)
        return {"answer": answer, "declined": answer.startswith("I don't know"), "chunks": chunks,
                "confidence": confidence, "latency": time.time() - t0}
