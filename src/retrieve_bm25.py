import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.json")
TOP_K = 5


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# Load the chunks
with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

# BM25 needs each chunk represented as a list of tokens
tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]

# Build the search index in memory
bm25 = BM25Okapi(tokenized_chunks)

print(f"Loaded {len(chunks)} chunks. BM25 search is ready.")

while True:
    question = input("\nAsk a Python question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break

    query_tokens = tokenize(question)

    if not query_tokens:
        print("Please enter a question containing words.")
        continue

    # Score every chunk against the question
    scores = bm25.get_scores(query_tokens)

    # Select the indices of the 5 highest-scoring chunks
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
        print(f"\n--- Result {rank} | BM25 score: {scores[index]:.2f} ---")
        print("Title:", chunk["title"])
        print("Source:", chunk["url"])
        print("Passage:", chunk["text"][:700])
