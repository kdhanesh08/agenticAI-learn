import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from tools import load_faiss_retriever, generate_answer

load_dotenv()

# Define the state schema
class GraphState(TypedDict):
    question: str
    chunks: List[str]
    answer: str

# Ensure the API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
retriever = load_faiss_retriever("embeddings/gemini_embeddings.json")

# Node 1: Retrieve
def retrieve_node(state):
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    return {"question": query, "chunks": docs}

# Node 2: Generate answer
def generate_node(state):
    return {"answer": generate_answer(llm, state["question"], state["chunks"])}

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"question": query})
        print("\n🧠 Final Answer:")
        print(result["answer"])
