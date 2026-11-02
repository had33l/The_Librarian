import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Input: extracted documents
INPUT_FILE = Path("data/processed/documents.json")

# Output: smaller passages for retrieval
OUTPUT_FILE = Path("data/processed/chunks.json")

# Split documents into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

# Load the extracted corpus
with INPUT_FILE.open("r", encoding="utf-8") as file:
    documents = json.load(file)

chunks = []

for document in documents:
    pieces = splitter.split_text(document["text"])

    for index, piece in enumerate(pieces):
        chunks.append({
            "chunk_id": f'{document["doc_id"]}_{index}',
            "doc_id": document["doc_id"],
            "chunk_index": index,
            "title": document["title"],
            "url": document["url"],
            "text": piece,
        })

# Save chunks as JSON
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Documents processed:", len(documents))
print("Chunks created:", len(chunks))
print("Average chunk characters:",
      round(sum(len(c["text"]) for c in chunks) / len(chunks), 1))
print("Saved to:", OUTPUT_FILE)
print("\nFirst chunk preview:")
print("Title:", chunks[0]["title"])
print("URL:", chunks[0]["url"])
print("Text:", chunks[0]["text"][:300])
