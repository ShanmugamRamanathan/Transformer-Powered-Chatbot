from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

def load_index(index_path="faiss.index", chunks_path="chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query, index, chunks, model_name="all-mpnet-base-v2", top_k=3):
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results

if __name__ == "__main__":
    index, chunks = load_index()
    query = input("Enter your question: ")
    results = search(query, index, chunks)
    print("\nTop relevant chunks:")
    for i, res in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(res)
        print()