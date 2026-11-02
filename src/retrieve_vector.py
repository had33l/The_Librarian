import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_FILE = Path("indexes/chunk_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Load the passages that we want to search
with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load the embedding model; the first run downloads it
model = SentenceTransformer(MODEL_NAME)

# Reuse saved embeddings when available; otherwise build them
if INDEX_FILE.exists():
    embeddings = np.load(INDEX_FILE)
    if len(embeddings) != len(chunks):
        raise ValueError("Index and chunks differ. Rebuild the index.")
    print("Loaded saved embeddings.")
else:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_FILE, embeddings)
    print("Saved embeddings to", INDEX_FILE)

print(f"Vector search ready: {len(chunks)} chunks.")

while True:
    question = input("\nAsk a Python question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break
    if not question:
        continue

    # Convert the question into a vector using the same model
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Compare the question vector with every chunk vector
    scores = embeddings @ query_vector

    # Select the indices of the five most similar chunks
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        print(f"\n--- Result {rank} | Similarity: {scores[index]:.3f} ---")
        print("Title:", chunk["title"])
        print("Source:", chunk["url"])
        print("Passage:", chunk["text"][:700])
