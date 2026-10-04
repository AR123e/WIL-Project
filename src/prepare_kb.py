"""
Chunk the raw KB markdown files into retrieval units.

Strategy
- FIDE rules: one chunk per numbered rule (3.7.3, 5.2.1 ...). Sub-clauses stay attached to their parent rule and
  heading-only lines ("3.7 The pawn:") are merged into the rule that follows.
- Glossaries: one chunk per term.
- Prose documents: one chunk per paragraph; very short heading-like paragraphs are merged into the next one.
- Every chunk keeps doc_id, doc_title, section, source and license so answers can cite and attribute them.
- Chunks of the beginner overview carry the tag "overview"; the "What is chess?" section is also flagged
  `intro`, and the quick-start section is flagged `howto`. The retriever uses these tags to answer whole-game
  questions such as "What is chess?" and "How do I play chess?".
"""
import json
import re

from src.config import RAW_DIR, CHUNKS_PATH

DOCS = [
    {"doc_id": "fide", "path": RAW_DIR / "fide_basic_rules.md", "kind": "rules",
     "title": "FIDE Laws of Chess - Basic Rules of Play",
     "source": "https://handbook.fide.com/chapter/e012023", "license": "FIDE (quoted for education)"},
    {"doc_id": "wikibooks", "path": RAW_DIR / "wikibooks_opening_principles.md", "kind": "prose",
     "title": "Chess Opening Principles (Wikibooks)",
     "source": "https://en.wikibooks.org/wiki/Chess/Basic_Openings", "license": "CC BY-SA"},
    {"doc_id": "overview", "path": RAW_DIR / "beginner_overview.md", "kind": "prose", "tags": ["overview"],
     "intro_sections": ["what is chess?"], "howto_sections": ["how to play chess: the quick version"],
     "title": "Chess for Absolute Beginners (project-authored)",
     "source": "project-authored", "license": "own work"},
    {"doc_id": "terms", "path": RAW_DIR / "tactics_and_terms.md", "kind": "prose",
     "title": "Chess Tactics and Common Terms (project-authored)",
     "source": "project-authored", "license": "own work"},
    {"doc_id": "strategy", "path": RAW_DIR / "strategy_basics.md", "kind": "prose",
     "title": "Basic Chess Strategy for Beginners (project-authored)",
     "source": "project-authored", "license": "own work"},
]

RULE_START = re.compile(r"^\d+(\.\d+)+")        # 1.1  3.7.3.1-2  5.2.1
MIN_LEN = 60


def split_sections(text):
    """Return [(header, body)] for every '## ' section; the title/Source preamble is dropped."""
    out = []
    for part in re.split(r"\n(?=##\s)", "\n" + text.strip()):
        part = part.strip()
        if not part.startswith("##"):
            continue
        head, _, body = part.partition("\n")
        header = head.lstrip("#").strip()
        body = "\n".join(l for l in body.split("\n") if not l.startswith("Source:")
                         and not l.startswith("http") and not l.startswith("(Basic Rules")).strip()
        if body:
            out.append((header, body))
    return out


def merge_short(blocks):
    """Merge heading-like short blocks into the block that follows."""
    merged, carry = [], ""
    for b in blocks:
        b = (carry + " " + b).strip()
        if len(b) < MIN_LEN:
            carry = b
        else:
            merged.append(b)
            carry = ""
    if carry and merged:
        merged[-1] += " " + carry
    return merged


def rule_blocks(body):
    blocks = []
    for line in (l.strip() for l in body.split("\n")):
        if not line:
            continue
        line = re.sub(r"\s{2,}", " ", line.replace("\t", " "))
        if RULE_START.match(line) or not blocks:
            blocks.append(line)
        else:
            blocks[-1] += " " + line
    return merge_short(blocks)


def prose_blocks(body):
    return merge_short([re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body) if p.strip()])


def build_chunks():
    chunks = []
    for doc in DOCS:
        text = doc["path"].read_text(encoding="utf-8")
        n = 0
        for header, body in split_sections(text):
            if header.lower().startswith("note on scope"):   # author's meta-note, not knowledge
                continue
            if header.lower().startswith("glossary"):
                blocks = [l.strip() for l in body.split("\n") if l.strip()]
            elif doc["kind"] == "rules":
                blocks = rule_blocks(body)
            else:
                blocks = prose_blocks(body)
            is_intro = header.lower() in doc.get("intro_sections", [])
            is_howto = header.lower() in doc.get("howto_sections", [])
            for b in blocks:
                n += 1
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_{n:03d}",
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["title"],
                    "section": header,
                    "text": b,
                    "source": doc["source"],
                    "license": doc["license"],
                    "tags": list(doc.get("tags", [])) + (["intro"] if is_intro else []) + (["howto"] if is_howto else []),
                })
    return chunks


def main():
    chunks = build_chunks()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    per_doc = {}
    for c in chunks:
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}  {per_doc}")


if __name__ == "__main__":
    main()
