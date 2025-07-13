import json
import faiss
import numpy as np
from langchain_core.documents import Document

def load_faiss_retriever(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    embeddings = []
    documents = []
    for item in raw:
        vec = item["embedding"]
        path = item["file"]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        embeddings.append(vec)
        documents.append(Document(page_content=content, metadata={"source": path}))

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    class StaticRetriever:
        def get_relevant_documents(self, query):
            return documents[:3]  # Simplified for now

    return StaticRetriever()

def generate_answer(llm, question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.invoke(prompt).content
