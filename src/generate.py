"""
Generation component: calls a locally running Ollama model with the retrieved chunks as context.
Answers must cite the chunk ids they used, so source attribution can be evaluated.
"""
import requests

from src.config import OLLAMA_URL, MODEL, JUDGE_MODEL

DECLINE = "I don't know based on the available information."

SYSTEM_PROMPT = f"""You are a friendly chess tutor for complete beginners.

Rules:
1. Answer using ONLY the context below. Do not use outside knowledge.
2. If the context only partly answers the question, explain the part it covers.
3. After each fact, cite the chunk id it came from in square brackets, e.g. [fide_016].
4. If the context does not contain the answer, reply exactly: {DECLINE}
5. Keep it short (max 4 sentences) and use simple words."""


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def _call(prompt, model=MODEL, temperature=0.0, timeout=120):
    r = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def generate_answer(question, chunks):
    return _call(build_prompt(question, chunks))


def generate_baseline(question):
    """No-retrieval baseline: the same LLM answering from its own knowledge (for the RAG-vs-no-RAG comparison)."""
    return _call("You are a chess tutor for beginners. Answer in at most 4 simple sentences.\n\nQuestion: "
                 + question + "\nAnswer:")


def judge_faithfulness(answer, chunks):
    """LLM-as-judge: is every claim in the answer supported by the context? Returns True/False."""
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = (
        "You are a strict fact-checker. Read the CONTEXT and the ANSWER.\n"
        "Reply with exactly one word: SUPPORTED if every factual claim in the ANSWER is stated in or directly "
        "implied by the CONTEXT, otherwise UNSUPPORTED.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nVerdict:"
    )
    verdict = _call(prompt, model=JUDGE_MODEL).upper()
    return verdict.startswith("SUPPORTED")
