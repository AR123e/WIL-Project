"""
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"

def _tokensize(text: str):
	return re.findall(r"[a-z0-9]+", text.lower())
class ChessRetriever:
	def __init__(self, chunks_path: Path = CHUNKS_PATH):
		self.chunks = json.loads(Path(chunks_path).read_text())
		corpus = [_tokensize(c["text"]) for c in self.chunks]
		self.bm25 = BM25Okapi(corpus)
	def retrieve(self, query: str, k: int = 3):
		scores = self.bm25.get_scores(_tokensize(query))
		ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
		results=[]
		for i in ranked:
			results.append({**self.chunks[i], "score": float(scores[i])})
		return results
if __name__=="__main__":
	retriever = ChessRetriever()
	demo_q = "How does a pawn capture another piece?"
	for r in retriever.retrieve(demo_q, k=3):
		print(f"[{r['score']:.2f}] {r['chunk_id']} ({r['section']})")
		print(" ", r["text"][:100].replace("
"," "))
