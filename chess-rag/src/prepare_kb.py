"""
 chunk the raw KB markdownvfiles into retrieval units
"""
import json
import re
from pathlib import Path
RAW_DIR = Path(__file__).parent.parent/"data"/"raw"
OUT_PATH = Path(__file__).parent.parent/"data"/"processed"/"chunks.json" 
DOCS = [
      {
	"doc_id": "fide_basic_rules",
	"path": RAW_DIR/ "fide_basic_rules.md",
	"title": "FIDE Laws of Chess - Basic Rules of play",
      },
      {
	"doc_id": "wikibooks_opening_principles",
	"path": RAW_DIR/ "wikibooks_opening_principles.md",
	"title": "Chess Opening Principles (Wikibooks)",
      },
]
def split_sections(text: str):
	parts = re.split(r"\n(?=##)", text)
	sections=[]
	for part in parts:
		part=part.strip()
		if not part or part.startswith("# "):
			if part.startswith("## "):
				pass
			else:
				continue
		lines = part.split("\n",1)
		header = lines[0].lstrip("#").strip()
		body = lines[1].strip() if len(lines)>1 else ""
		if body:
			sections.append((header,body))
	return sections

def split_paragraphs(body: str):
	paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
	return paras

def main():
	chunks=[]
	chunk_counter = 0
	for doc in DOCS:
		text = doc["path"].read_text(encoding="utf-8")
		sections = split_sections(text)
		for section_title, section_body in sections:
			paragraphs = split_paragraphs(section_body)
			for para in paragraphs:
				if len(para)<40:
					continue
				chunk_counter+=1
				chunks.append(
					{
						"chunk_id": f"{doc['doc_id']}_{chunk_counter:03d}",
						"doc_id": doc["doc_id"],
						"doc_title": doc["title"],
						"section": section_title,
						"text": para,	
					}
				) 
	OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUT_PATH.write_text(json.dumps(chunks, indent=2),encoding="utf-8")
	print(f"Wrote {len(chunks)} chunks to {OUT_PATH}")
if __name__ == "__main__":
	main() 
