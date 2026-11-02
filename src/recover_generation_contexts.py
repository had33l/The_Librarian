import json
from pathlib import Path
import argparse

from src.ask import load_retriever, retrieve

QUESTIONS_FILE = Path("data/eval_questions.json")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval/generation_results_openai.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/generation_contexts_openai.json"),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists. Choose another output path."
        )

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))

    if len(questions) != len(results):
        raise ValueError("Question and generation result counts differ.")

    print("Loading retriever...")
    chunks, embeddings, bm25, model = load_retriever()

    contexts = []

    for index, (question, result) in enumerate(
        zip(questions, results), start=1
    ):
        if question["question"] != result["question"]:
            raise ValueError(f"Question mismatch at index {index}")

        retrieved_chunks, _ = retrieve(
            question["question"], chunks, embeddings, bm25, model
        )

        contexts.append({
            "id": result["id"],
            "question": question["question"],
            "retrieved_chunks": retrieved_chunks,
            "retrieved_sources": [
                chunk["url"] for chunk in retrieved_chunks
            ],
        })

        if index % 10 == 0:
            print(f"Recovered {index}/{len(questions)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contexts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matching = sum(
        context["retrieved_sources"] == result["retrieved_sources"]
        for context, result in zip(contexts, results)
    )

    print("\nRecovery complete.")
    print(f"Contexts saved: {len(contexts)}")
    print(f"Source lists matching original results: {matching}/100")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()