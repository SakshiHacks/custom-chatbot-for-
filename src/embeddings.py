from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return np.array(embeddings)

def embed_query(query):
    return model.encode([query])