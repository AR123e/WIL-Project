"""Single source of truth for settings shared by pipeline, app and evaluation."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EVAL_DIR = ROOT / "eval"

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE}/api/tags"
MODEL = os.getenv("CHESS_MODEL", "llama3.1")             # answer generator
JUDGE_MODEL = os.getenv("CHESS_JUDGE_MODEL", MODEL)       # LLM-as-judge (use a larger model if you have one)

# ---- retrieval -----------------------------------------------------------------------------
RETRIEVER = os.getenv("CHESS_RETRIEVER", "hybrid")        # "bm25" or "hybrid" (BM25 + char n-gram TF-IDF, RRF fusion)
TOP_K = 3                                                 # chunks passed to the LLM
RRF_K = 60                                                # reciprocal-rank-fusion constant

# ---- confidence gate (tuned on the DEV split only, see src/calibrate.py) --------------------------
MIN_SCORE = 2.0            # best BM25 score among retrieved chunks
MIN_CHAR_SIM = 0.20        # best char n-gram cosine similarity among retrieved chunks
MIN_COVERAGE = 0.0         # share of query content words found in the top chunk (off unless calibration says otherwise)
MIN_WCOVERAGE = 0.0        # rarity-weighted coverage of the top chunk (off unless calibration says otherwise)
MAX_OOV = 0.45             # decline if more than this share of query words never occur in the KB (dev: feasible range 0.40-0.50)

# ---- generation ----------------------------------------------------------------------------
STRICT_CITATIONS = False   # True: answers without a valid [chunk_id] citation are replaced by the decline message
