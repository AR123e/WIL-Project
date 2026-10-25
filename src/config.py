"""Single source of truth for settings shared by pipeline, app and evaluation."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EVAL_DIR = ROOT / "eval"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL="http://localhost:11434/api/tags"
MODEL = "llama3.1"          # answer generator
JUDGE_MODEL = "llama3.1"    # LLM-as-judge for faithfulness (swap for a bigger model if available)

TOP_K = 3                   # chunks passed to the LLM
MIN_SCORE = 2.0             # BM25 top-1 score below this => decline (see src/calibrate.py)
MIN_COVERAGE = 0.3          # optional second gate; calibration showed it does not separate in/out-of-scope, so off
BM25_K1=0.6
BM25_B=0.75
MAX_QUESTIONS=5
