# ♟️ Chess Mentor – RAG for Beginner Chess Learners

A simple Retrieval-Augmented Generation (RAG) app that helps beginners learn chess.

The system:
- Retrieves relevant info from a small chess knowledge base
- Uses a local LLM (Ollama - Llama 3.1) to generate answers
- Shows sources (chunks) and avoids guessing ("I don't know" when unsure)

Everything runs locally — no paid APIs.

---

## 👥 Team Members

| Student ID | Name |
|-----------|------|
| s4205613 | Alen Regi |
| s4215559 | Delona Sebastian |
| s4216144 | Chayanika Bangar |
| s4219934 | Goutham Thilak |
| s4186078 | Anmol Kaushil |

---

## 🧠 How It Works

User Question  
↓  
Retriever (BM25 + n-grams)  
↓  
Top relevant chunks  
↓  
LLM (Ollama - Llama 3.1)  
↓  
Answer + Sources  
↓  
Streamlit UI  

---

## 📂 Project Structure
chess-rag/
│
├── app.py # Streamlit UI
├── src/
│ ├── pipeline.py # Main RAG pipeline
│ ├── retrieve.py # Retrieval logic (BM25)
│ ├── generate.py # LLM response (Ollama)
│ ├── prepare_kb.py # Chunking
│ ├── evaluate.py # Evaluation
│
├── data/
│ ├── raw/ # Original documents
│ └── processed/
│ └── chunks.json # Chunked data
│
├── eval/
│ ├── test_questions.json
│ └── results.csv
│
└── tests/ # Unit tests

---

## 🚀 How to Run (Windows & Mac)

### 1. Install dependencies

**Windows**
py -m pip install -r requirements.txt

**Mac / Linux**

---

### 2. Prepare knowledge base
py src/prepare_kb.py # Windows
python3 src/prepare_kb.py # Mac


---

### 3. Start Ollama (LLM)

First install Ollama: https://ollama.com/

Then run:
ollama pull llama3.1
#### ▶️ Run model
ollama run llama3.1

#### ⚠️ If GPU error (Windows fix)

$env:OLLAMA_NO_GPU=1
ollama run llama3.1


(For Mac, GPU usually works automatically)

---

### 4. Run the app
py -m streamlit run app.py # Windows
python3 -m streamlit run app.py # Mac


Open: http://localhost:8501

---

## 📊 Evaluation

We evaluate the system using:

- Hit@1 (top result correct)
- Hit@3 (top 3 results contain answer)
- Unanswered rate (confidence-based)

Run:
python src/evaluate.py

Results:eval/results.csv\

---

## ⚠️ Limitations

- Uses BM25 (keyword-based), not semantic search
- Small knowledge base → limited answers
- Sometimes returns "I don't know"
- LLM (Ollama) may fail if not running

---

## 🔮 Future Improvements

- Add embeddings (vector database)
- Improve chunking strategy
- Expand knowledge base
- Improve answer quality

---

## 🎯 Key Features

- Runs fully locally (no API cost)
- Shows retrieved sources (transparent answers)
- Avoids hallucination
- Interactive UI

---

## 📌 Conclusion

This project demonstrates a simple but complete RAG system:

- Retrieval (BM25)
- Generation (LLM)
- Evaluation (metrics + testing)
