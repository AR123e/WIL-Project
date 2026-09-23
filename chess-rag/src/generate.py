"""
Generation component of the RAG pipeline.
Calls a locally-running Ollama model with retrieved chunks as context.
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"

SYSTEM_PROMPT = """You are a friendly chess assistant for absolute beginners.
Answer ONLY using the provided context. If the context does not contain the
answer, say you don't know rather than guessing. Keep answers short and
plain-language."""


def build_prompt(question, chunks):
    context_parts = []
    for c in chunks:
        context_parts.append(c["chunk_id"] + ": " + c["text"])
    context = "\n\n".join(context_parts)

    prompt = SYSTEM_PROMPT + "\n\nContext:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"
    return prompt


def generate_answer(question, chunks):
    prompt = build_prompt(question, chunks)
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()