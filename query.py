import os, sys, subprocess
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.expanduser("~/rag-test/chroma_db")
LLAMA_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODEL_PATH = os.path.expanduser("~/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")  # swap when you upgrade

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("docs")

query = " ".join(sys.argv[1:])
query_embedding = model.encode(query).tolist()

results = collection.query(query_embeddings=[query_embedding], n_results=3)
context = "\n\n".join(results["documents"][0])

prompt = f"""Use the context below to answer the question. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}
Answer:"""

print("=== Retrieved context ===")
print(context)
print("\n=== Generating ===\n")

with open("/tmp/rag_prompt.txt", "w") as f:
    f.write(prompt)

subprocess.run([LLAMA_BIN, "-m", MODEL_PATH, "-f", "/tmp/rag_prompt.txt", "-n", "200"])
