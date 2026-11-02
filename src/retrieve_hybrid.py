import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_FILE = Path("indexes/chunk_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# Prepare keyword search.
tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Load the same embedding model and saved vectors.
model = SentenceTransformer(MODEL_NAME)
embeddings = np.load(INDEX_FILE)

if len(embeddings) != len(chunks):
    raise ValueError("Index and chunks differ. Rebuild the index.")

print(f"Hybrid search ready: {len(chunks)} chunks.")

while True:
    question = input("\nAsk a Python question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break
    if not question:
        continue

    # Rank passages by keyword matching.
    keyword_scores = bm25.get_scores(tokenize(question))
    keyword_ranking = np.argsort(keyword_scores)[::-1]

    # Rank passages by meaning.
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    vector_scores = embeddings @ query_vector
    vector_ranking = np.argsort(vector_scores)[::-1]

    # Combine both rankings using RRF.
    rrf_scores = {}
    for ranking in (keyword_ranking, vector_ranking):
        for rank, index in enumerate(ranking, start=1):
            rrf_scores[int(index)] = (
                rrf_scores.get(int(index), 0)
                + 1 / (RRF_K + rank)
            )

    top_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:TOP_K]

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
        print(f"\n--- Result {rank} | RRF score: {rrf_scores[index]:.4f} ---")
        print("Title:", chunk["title"])
        print("Source:", chunk["url"])
        print("Passage:", chunk["text"][:700])