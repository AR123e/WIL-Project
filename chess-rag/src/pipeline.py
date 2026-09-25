from src.retrieve import ChessRetriever
from src.generate import generate_answer

class ChessRAG:
    def __init__(self):
        self.retriever = ChessRetriever()

    def ask(self, question, k=5):
        chunks = self.retriever.retrieve(question, k=k)

        # Confidence check (same as eval)
        if not chunks or chunks[0]["score"] < 5.0:
            return {
                "answer": "I don't know based on the available information.",
                "chunks": chunks,
                "confidence": chunks[0]["score"] if chunks else 0
            }

        answer = generate_answer(question, chunks)

        return {
            "answer": answer,
            "chunks": chunks,
            "confidence": chunks[0]["score"]
        }