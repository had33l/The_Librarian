import argparse
import csv
import json
import re
import time
from pathlib import Path

from src.ask import (
    load_retriever,
    retrieve,
    generate_answer,
    GENERATION_MODEL,
)

QUESTIONS_FILE = Path("data/eval_questions.json")
OUTPUT_JSON = Path("eval/generation_results.json")
OUTPUT_CSV = Path("eval/generation_results.csv")

ABSTENTION = (
    "I don't know based on the available Python documentation."
)


def get_expected_sources(question):
    sources = question.get("expected_sources")

    if sources is None:
        sources = question.get("expected_source", [])

    if isinstance(sources, str):
        sources = [sources]

    return sources


def evaluate_citations(answer, retrieved_chunks, expected_sources):
    cited_ids = set(re.findall(r"\[S(\d+)\]", answer))

    valid_ids = {
        str(i) for i in range(1, len(retrieved_chunks) + 1)
    }

    valid_citations = cited_ids & valid_ids
    invalid_citations = cited_ids - valid_ids

    cited_urls = [
        retrieved_chunks[int(source_id) - 1]["url"]
        for source_id in valid_citations
    ]

    expected_source_cited = any(
        source in cited_urls for source in expected_sources
    )

    return {
        "citation_count": len(cited_ids),
        "valid_citation_count": len(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "expected_source_cited": expected_source_cited,
        "cited_urls": cited_urls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all).",
    )
    parser.add_argument(
        "--model",
        default=GENERATION_MODEL,
        help="Generation model to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=OUTPUT_CSV,
    )
    args = parser.parse_args()
    if args.output.exists() or args.csv.exists():
        raise FileExistsError(
            "Output file already exists. Choose different filenames."
        )

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if args.limit:
        questions = questions[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading RAG system...")
    chunks, embeddings, bm25, model = load_retriever()

    print(f"Evaluating {len(questions)} questions...\n")

    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = get_expected_sources(item)

        print(f"[{index}/{len(questions)}] {question}")

        try:
            retrieved_chunks, retrieval_ms = retrieve(
                question, chunks, embeddings, bm25, model
            )

            answer, generation_ms, usage = generate_answer(
                question, retrieved_chunks, model=args.model
            )

            citation_metrics = evaluate_citations(
                answer, retrieved_chunks, expected_sources
            )

            result = {
                "id": item.get("id", index),
                "question": question,
                "expected_answer": item.get("expected_answer", ""),
                "supporting_passage": item.get(
                    "supporting_passage", ""
                ),

                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],


                "expected_sources": expected_sources,
                "answer": answer,
                "retrieved_sources": [
                    chunk["url"] for chunk in retrieved_chunks
                ],
                **citation_metrics,
                "abstained": ABSTENTION.lower() in answer.lower(),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(retrieval_ms + generation_ms, 2),
                "error": None,
            }

        except Exception as error:
            result = {
                "id": item.get("id", index),
                "question": question,
                "error": str(error),
            }

        results.append(result)

        # Save after every question so progress isn't lost.
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        print(f"   Answer: {result.get('answer', result.get('error'))}")
        print()

    # Export CSV for easier analysis.
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    successful = [r for r in results if not r.get("error")]

    if successful:
        cited = [r for r in successful if r["citation_count"] > 0]
        expected_cited = sum(
            r["expected_source_cited"] for r in successful
        )
        abstentions = sum(r["abstained"] for r in successful)

        print("\n========== GENERATION EVALUATION ==========")
        print(f"Questions evaluated: {len(successful)}")
        print(
            f"Answers with citations: "
            f"{len(cited)}/{len(successful)}"
        )
        print(
            f"Expected source cited: "
            f"{expected_cited}/{len(successful)} "
            f"({expected_cited / len(successful):.1%})"
        )
        print(f"Abstentions: {abstentions}")

        print(
            f"Average generation latency: "
            f"{sum(r['generation_ms'] for r in successful) / len(successful):.1f} ms"
        )
        print(
            f"Average total latency: "
            f"{sum(r['total_ms'] for r in successful) / len(successful):.1f} ms"
        )

    print(f"\nSaved results to {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
