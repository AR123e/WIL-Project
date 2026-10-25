# Chess Mentor – RAG for beginner chess learners

**Domain / stakeholder:** beginner chess players (and coaches) who need rule and opening answers they can trust.
**Value claim:** answers are grounded in a curated knowledge base (FIDE Laws of Chess, Wikibooks opening principles and three plain-language beginner guides), cite them, and the system declines rather than guesses.
**Stack:** BM25 (rank_bm25) → Ollama `llama3.1` → Streamlit. No paid APIs.

## Setup (macOS)
```bash
cd WIL-Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
brew install ollama            # skip if installed
ollama serve & ollama pull llama3.1
```

## Setup (Windows)
Use PowerShell (or Windows Terminal).
```powershell
cd WIL-Project
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
winget install Ollama.Ollama   # or download the installer from https://ollama.com/download
ollama pull llama3.1
```


## Run
```bash
python -m src.prepare_kb                          # build data/processed/chunks.json
python -m pytest -q
python -m src.calibrate                           # inspect score thresholds (optional)
python -m src.evaluate --stage retrieval          # fast, no LLM
python -m src.evaluate --stage full --baseline    # full framework + no-RAG baseline
streamlit run app.py                              # demo app
```

## Layout
```
data/raw/            fide_basic_rules.md, wikibooks_opening_principles.md, beginner_overview.md,
startegy_basics.md, tactics_and_terms.md
data/processed/      chunks.json (generated)
eval/                test_questions.json (32 Qs: 12 known, 12 inferred, 8 out_of_scope), results_*.csv
src/config.py        all settings (model, k, thresholds)
src/prepare_kb.py    rule-aware chunking (one chunk per FIDE rule / glossary term / paragraph)
src/retrieve.py      BM25 + coverage signal + routing of "What is chess?"-style questions
src/generate.py      Ollama call, cited answers, no-RAG baseline, LLM faithfulness judge
src/pipeline.py      retrieve -> confidence gate -> generate
src/evaluate.py      evaluation framework
app.py               Streamlit UI + evaluation dashboard
tests/               pytest regression tests
 .github/workflows/  CI: runs tests on every push
```

