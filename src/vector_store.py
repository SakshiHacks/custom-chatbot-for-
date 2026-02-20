import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def search(index, query_vector, top_k=5):
    query_vector = query_vector.astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)
    return distances, indices