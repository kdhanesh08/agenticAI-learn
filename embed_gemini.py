import os
import json
import glob
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

os.makedirs("embeddings", exist_ok=True)

def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response["embedding"]

chunk_files = glob.glob("chunks/*.txt")
all_embeddings = []

for file_path in chunk_files:
    with open(file_path, "r", encoding="utf-8") as f:
        chunk = f.read().strip()
        if not chunk or len(chunk.split()) < 5:
            print(f"⚠️ Skipping short chunk: {file_path}")
            continue
        embedding = get_embedding(chunk)
        all_embeddings.append({"file": file_path, "embedding": embedding})

with open("embeddings/gemini_embeddings.json", "w") as f:
    json.dump(all_embeddings, f)

print(f"✅ Embedded {len(all_embeddings)} chunks using Gemini.")