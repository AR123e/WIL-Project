import json
from collections import Counter
from pathlib import Path
chunks = json.loads(Path("data/processed/chunks.json").read_text())
print("Counts per doc_id:", Counter(c["doc_id"] for c in chunks))
print()
for c in chunks:
	print(c["chunk_id"],"|", c["text"][:60].replace("\n", " "))