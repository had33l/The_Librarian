import json
import random
import re
from pathlib import Path

import ollama

DOCUMENTS_FILE = Path("data/processed/documents.json")
OUTPUT_FILE = Path("data/eval_questions.json")

MODEL = "qwen2.5:3b"
TARGET_QUESTIONS = 100
random.seed(42)


def generate_qa(document):
    passage = document["text"][:2500]

    prompt = f"""
You are creating an evaluation question for a Python documentation RAG system.

Use ONLY the documentation passage below.

Generate ONE specific, useful question that can be answered
directly from this passage.

Return ONLY valid JSON with these fields:
{{
  "question": "A natural question",
  "expected_answer": "A concise, accurate answer"
}}

Do not ask about information absent from the passage.
Avoid questions whose answer is simply the document title.

DOCUMENT:
{document["title"]}

PASSAGE:
{passage}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 250}
    )

    content = response.message.content.strip()
    content = re.sub(r"^```json\s*|^```\s*|\s*```$", "", content)

    return json.loads(content)


def main():
    documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))

    # Select sufficiently substantial documents.
    candidates = [
        doc for doc in documents
        if len(doc["text"]) >= 700
    ]

    random.shuffle(candidates)

    results = []
    seen_questions = set()

    for document in candidates:
        if len(results) >= TARGET_QUESTIONS:
            break

        try:
            qa = generate_qa(document)

            question = qa["question"].strip()
            answer = qa["expected_answer"].strip()

            if not question or not answer:
                continue

            normalized = question.lower()
            if normalized in seen_questions:
                continue

            results.append({
                "question": question,
                "expected_source": document["url"],
                "expected_answer": answer,
                "supporting_passage": document["text"][:2500]
            })

            seen_questions.add(normalized)

            print(f"[{len(results)}/{TARGET_QUESTIONS}] {question}")

        except Exception as error:
            print(f"Skipped {document['doc_id']}: {error}")

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nGenerated {len(results)} questions.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
