import numpy as np

def rerank(query_vector, chunk_embeddings, indices):
    scores = []

    for idx in indices[0]:
        score = np.dot(query_vector[0], chunk_embeddings[idx])
        scores.append((score, idx))

    scores.sort(reverse=True)
    return scores