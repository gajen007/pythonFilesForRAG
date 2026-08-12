import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.expanduser("~/rag-test/docs")
DB_DIR = os.path.expanduser("~/rag-test/chroma_db")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("docs")

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

doc_id = 0
for fname in os.listdir(DOCS_DIR):
    path = os.path.join(DOCS_DIR, fname)
    with open(path) as f:
        text = f.read()
    for chunk in chunk_text(text):
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[str(doc_id)],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": fname}]
        )
        doc_id += 1

print(f"Ingested {doc_id} chunks from {DOCS_DIR}")