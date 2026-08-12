import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.expanduser("~/rag-test/chroma_db")

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("docs")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3

@app.post("/retrieve")
def retrieve(req: QueryRequest):
    query_embedding = model.encode(req.query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=req.n_results
    )
    return {
        "query": req.query,
        "chunks": results["documents"][0],
        "sources": [m.get("source") for m in results["metadatas"][0]]
    }