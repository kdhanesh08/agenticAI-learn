import numpy as np
import json
import faiss
import os
from embed_gemini import get_embedding

def load_embeddings(path="embeddings/gemini_embeddings.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = []
    metadata = []
    for item in data:
        embeddings.append(item["embedding"])
        metadata.append(item["file"])
    return np.array(embeddings).astype("float32"), metadata

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(query, embeddings, metadata, index, top_k=3):
    query_vector = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        with open(metadata[idx], "r", encoding="utf-8") as f:
            content = f.read()
        results.append({
            "chunk": os.path.basename(metadata[idx]),
            "score": float(distances[0][i]),
            "content": content.strip()
        })
    return results

if __name__ == "__main__":
    embeddings, metadata = load_embeddings()
    index = build_index(embeddings)

    while True:
        query = input("\n🔎 Ask your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        results = search(query, embeddings, metadata, index)
        for i, res in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"📄 Chunk: {res['chunk']}")
            print(f"📉 Score: {res['score']:.2f}")
            print("📘 Content Preview:")
            print(res["content"][:300])