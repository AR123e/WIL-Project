# ♟️ Chess Mentor RAG System  
**WIL Project – Group 52**

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

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers beginner-level chess questions using a knowledge base.

The system:
- Retrieves relevant information from chess documents
- Uses a local Large Language Model (LLM) to generate answers
- Avoids hallucination by answering only from retrieved context

---

## 🧠 System Architecture
User Question
↓
BM25 Retriever (retrieve.py)
↓
Top-K Relevant Chunks
↓
LLM (Ollama - Llama3.1)
↓
Generated Answer
↓
Streamlit UI


---

## ⚙️ Technologies Used

- Python  
- Streamlit (UI)  
- BM25 (rank_bm25) – Retrieval  
- Ollama (Llama 3.1) – Local LLM  
- Pandas – Evaluation  

---

## 📂 Project Structure
chess-rag/
│
├── app.py # Streamlit UI
├── src/
│ ├── pipeline.py # Main RAG pipeline
│ ├── retrieve.py # BM25 retrieval
│ ├── generate.py # LLM generation (Ollama)
│ ├── prepare_kb.py # Chunking logic
│ └── evaluate.py # Evaluation script
│
├── data/
│ ├── raw/ # Raw markdown documents
│ └── processed/
│ └── chunks.json # Chunked knowledge base
│
├── eval/
│ └── results.csv # Evaluation results


---

## 🚀 How to Run the Project

### 1. Install dependencies
pip install -r requirements.txt

---

### 2. Prepare knowledge base
py src/prepare_kb.py

---

### 3. Start Ollama (LLM)
ollama run llama3.1
*If GPU error occurs:
  $env:OLLAMA_NO_GPU=1
  ollama run llama3.1

---

### 4. Run the application
py -m streamlit run app.py

Open in browser:http://localhost:8501

---

## 📊 Evaluation

The system is evaluated using:

- Hit@1  
- Hit@3  
- Unanswered Rate  

Run evaluation:
python src/evaluate.py


Results are saved in: eval/results.csv


---

## ⚠️ Limitations

- Limited knowledge base (only beginner chess concepts)  
- Retrieval depends on chunk quality  
- Cannot answer questions outside provided documents  

---

## 🔮 Future Improvements

- Add more chess documents  
- Use semantic search (embeddings)  
- Improve chunking strategy  
- Add conversational memory  

---

## 🎯 Key Features

- Local LLM (no API cost)  
- Explainable answers (shows retrieved chunks)  
- Hallucination control (confidence-based)  
- Interactive Streamlit UI  

---

## 📌 Conclusion

This project demonstrates a complete **RAG pipeline**, combining:
- Information Retrieval  
- Natural Language Generation  
- Interactive UI  

It highlights how LLMs can be grounded using external knowledge to produce more reliable and explainable answers.

---
