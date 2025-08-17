from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load chunks from previous step
from chunk_document import load_text, chunk_text

def build_index(chunks, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def save_index(index, chunks, index_path="faiss.index", chunks_path="chunks.pkl"):
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    text = load_text("document2.txt")
    chunks = chunk_text(text)
    index, embeddings = build_index(chunks)
    save_index(index, chunks)
    print("FAISS index and chunks saved.")