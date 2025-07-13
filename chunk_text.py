import re
from pathlib import Path

with open("cleaned_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = re.split(r'(?<=[.!?]) +', text)
chunks = []
chunk = ""
for i, sentence in enumerate(sentences):
    chunk += sentence + " "
    if (i + 1) % 2 == 0 or i == len(sentences) - 1:
        chunks.append(chunk.strip())
        chunk = ""

chunk_dir = Path("chunks")
chunk_dir.mkdir(exist_ok=True)

for i, chunk in enumerate(chunks):
    with open(chunk_dir / f"chunk_{i+1}.txt", "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"✅ Saved {len(chunks)} chunks to 'chunks/' folder.")