import os
import numpy as np

from pdf_loader import extract_text_from_pdf
from chunking import chunk_text
from embeddings import create_embeddings, embed_query
from vector_store import create_faiss_index, search
from reranker import rerank
from generator import generate_answer

# ------------------------
# LOAD ALL PDFs
# ------------------------
print("Loading PDFs...")

all_text = ""

for file in os.listdir("../data"):
    if file.endswith(".pdf"):
        print(f"Loading {file}...")
        all_text += extract_text_from_pdf(f"../data/{file}")

print("Chunking text...")
chunks = chunk_text(all_text)

print("Creating embeddings...")
embeddings = create_embeddings(chunks)

print("Creating FAISS index...")
index = create_faiss_index(embeddings)

print("\nSystem Ready! Ask your questions (type 'exit' to quit)\n")

# ------------------------
# QA LOOP
# ------------------------
while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    query_vector = embed_query(query)

    distances, indices = search(index, np.array(query_vector))

    # Confidence score
    top_score = float(distances[0][0])

    if top_score < 0.55:
        print("\nAnswer not found in the document.")
        print(f"Confidence: {top_score:.2f}\n")
        continue

    # Re-rank
    ranked = rerank(query_vector, embeddings, indices)

    # Take best chunk
    top_indices = [idx for score, idx in ranked[:3]]

    context = "\n\n".join([chunks[i] for i in top_indices])

    answer = generate_answer(context, query)

    print("\nFinal Answer:\n")
    print(answer)
    print(f"\nConfidence: {top_score:.2f}\n")