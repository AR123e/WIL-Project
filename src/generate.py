"""
Generation component: builds the grounded prompt and calls the LLM. Answers must cite the chunk ids they used,
so source attribution can be checked in code and evaluated.
"""
import re

from src.config import MODEL, JUDGE_MODEL
from src.llm import OllamaLLM

DECLINE = "I don't know based on the available information."
CITE = re.compile(r"\[([a-z]+_\d{3})\]")

SYSTEM_PROMPT = f"""You are a friendly chess tutor for complete beginners.

Rules:
1. Answer using ONLY the context below. Do not use outside knowledge.
2. If the context only partly answers the question, explain the part it covers.
3. After each fact, cite the chunk id it came from in square brackets, e.g. [fide_016].
4. If the question asks what chess is or how to get started, give a short, friendly summary of the context.
5. If the context does not contain the answer, reply exactly: {DECLINE}
6. Keep it short (max 4 sentences) and use simple words."""

_llm = OllamaLLM()


def set_llm(llm):
    """Swap the LLM client (used by tests and notebooks)."""
    global _llm
    _llm = llm


def get_llm():
    return _llm


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def generate_answer(question, chunks):
    return _llm.generate(build_prompt(question, chunks), model=MODEL)


def generate_baseline(question):
    """No-retrieval baseline: the same LLM answering from its own knowledge (for the RAG-vs-no-RAG comparison)."""
    return _llm.generate("You are a chess tutor for beginners. Answer in at most 4 simple sentences.\n\nQuestion: "
                         + question + "\nAnswer:", model=MODEL)


def split_citations(answer, chunks):
    """Return (cited_ids_that_were_retrieved, invented_ids_that_were_not)."""
    cited = CITE.findall(answer)
    retrieved = {c["chunk_id"] for c in chunks}
    return [c for c in dict.fromkeys(cited) if c in retrieved], [c for c in dict.fromkeys(cited) if c not in retrieved]


def judge_faithfulness(answer, chunks):
    """LLM-as-judge: is every claim in the answer supported by the context? Returns True/False."""
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = (
        "You are a strict fact-checker. Read the CONTEXT and the ANSWER.\n"
        "Reply with exactly one word: SUPPORTED if every factual claim in the ANSWER is stated in or directly "
        "implied by the CONTEXT, otherwise UNSUPPORTED.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nVerdict:"
    )
    return _llm.generate(prompt, model=JUDGE_MODEL).upper().startswith("SUPPORTED")
