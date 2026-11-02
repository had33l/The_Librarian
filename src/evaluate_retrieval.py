import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.json")
QUESTIONS_FILE = Path("data/eval_questions.json")
INDEX_FILE = Path("indexes/chunk_embeddings.npy")
RESULTS_FILE = Path("eval/retrieval_results.json")
CSV_FILE = Path("eval/retrieval_results.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60
CANDIDATE_K = 100


def tokenize(text):
    """Split text into lowercase words for keyword search."""
    return re.findall(r"\b\w+\b", text.lower())

def top_candidate_ranking(scores, candidate_k=CANDIDATE_K):
    """Return the highest-scoring candidates in ranking order."""
    candidate_k = min(candidate_k, len(scores))

    # Find the candidate pool without fully sorting every chunk.
    candidates = np.argpartition(scores, -candidate_k)[-candidate_k:]

    # Sort only the candidates by descending score.
    ordered = candidates[np.argsort(scores[candidates])[::-1]]

    return ordered


def get_source(chunk):
    """Return the source page for a retrieved passage."""
    return chunk["url"].rstrip("/")


def reciprocal_rank(expected_sources, ranking, chunks):
    """
    Return 1/rank of the first expected source.
    Return 0 if no expected source appears in the ranking.
    """
    expected_sources = {
        source.rstrip("/") for source in expected_sources
    }

    for rank, index in enumerate(ranking, start=1):
        if get_source(chunks[int(index)]) in expected_sources:
            return 1.0 / rank

    return 0.0


def evaluate_question(
    question, expected_sources, chunks, bm25, model, embeddings
):
    expected = {source.rstrip("/") for source in expected_sources}

    def measure_retrieval(method):
        """Run one complete retrieval method and measure its latency."""
        start = time.perf_counter()

        if method == "bm25":
            scores = bm25.get_scores(tokenize(question))
            ranking = top_candidate_ranking(scores)

        elif method == "vector":
            query_vector = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            scores = embeddings @ query_vector
            ranking = top_candidate_ranking(scores)

        elif method == "hybrid":
            # Score all chunks with both retrieval methods.
            keyword_scores = bm25.get_scores(tokenize(question))

            query_vector = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            vector_scores = embeddings @ query_vector

            # Keep the best candidates from each method.
            keyword_ranking = top_candidate_ranking(keyword_scores)
            vector_ranking = top_candidate_ranking(vector_scores)

            # Fuse the candidate rankings using Reciprocal Rank Fusion.
            rrf_scores = {}

            for candidate_ranking in (
                keyword_ranking,
                vector_ranking,
            ):
                for rank, index in enumerate(
                    candidate_ranking, start=1
                ):
                    index = int(index)
                    rrf_scores[index] = (
                        rrf_scores.get(index, 0.0)
                        + 1.0 / (RRF_K + rank)
                    )

            ranking = sorted(
                rrf_scores,
                key=rrf_scores.get,
                reverse=True,
            )

        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        latency_ms = (time.perf_counter() - start) * 1000
        return ranking, latency_ms

    results = {}

    for method in ("bm25", "vector", "hybrid"):
        ranking, latency_ms = measure_retrieval(method)
        top_indices = [int(i) for i in ranking[:TOP_K]]

        top_sources = [
            get_source(chunks[i]) for i in top_indices
        ]

        recall = float(any(source in expected for source in top_sources))
        mrr = reciprocal_rank(expected_sources, top_indices, chunks)

        results[method] = {
            "recall_at_5": recall,
            "mrr_at_5": mrr,
            "latency_ms": round(latency_ms, 3),
            "top_results": [
                {
                    "title": chunks[i]["title"],
                    "url": chunks[i]["url"],
                    "chunk_id": chunks[i]["chunk_id"],
                }
                for i in top_indices
            ],
        }

    return results


def percentile(values, percentile_value):
    """Calculate a latency percentile."""
    return round(float(np.percentile(values, percentile_value)), 3)


def main():
    # Load the passages and evaluation questions.
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("The evaluation question file is empty.")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Missing vector index: {INDEX_FILE}. "
            "Run src/retrieve_vector.py first."
        )

    embeddings = np.load(INDEX_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The vector index and chunks do not match. "
            "Rebuild the embeddings."
        )

    # Prepare BM25 once; don't rebuild it for every question.
    print("Preparing BM25 keyword search...")
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Warm up the model before measuring query latency.
    # Model loading time is not included in per-question latency.
    model.encode(["warm up"], convert_to_numpy=True)

    all_results = []
    csv_rows = []

    print(f"\nEvaluating {len(questions)} questions...\n")

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        # Accept either one expected source or a list of sources.
        expected_sources = item.get(
            "expected_sources",
            item.get("expected_source"),
        )

        if isinstance(expected_sources, str):
            expected_sources = [expected_sources]

        if not expected_sources:
            raise ValueError(
                f"Question {number} has no expected source."
            )

        method_results = evaluate_question(
            question,
            expected_sources,
            chunks,
            bm25,
            model,
            embeddings,
        )

        record = {
            "question": question,
            "expected_sources": expected_sources,
            "methods": method_results,
        }
        all_results.append(record)

        print(f"{number}. {question}")
        for method, metrics in method_results.items():
            print(
                f"   {method:6} | "
                f"Recall@5={metrics['recall_at_5']:.0f} | "
                f"MRR@5={metrics['mrr_at_5']:.3f} | "
                f"Latency={metrics['latency_ms']:.3f} ms"
            )

            csv_rows.append({
                "question": question,
                "method": method,
                "recall_at_5": metrics["recall_at_5"],
                "mrr_at_5": metrics["mrr_at_5"],
                "latency_ms": metrics["latency_ms"],
                "top_5_urls": " | ".join(
                    result["url"]
                    for result in metrics["top_results"]
                ),
            })

    # Calculate averages across all questions.
    summary = {}

    for method in ("bm25", "vector", "hybrid"):
        recalls = [
            row["methods"][method]["recall_at_5"]
            for row in all_results
        ]
        mrrs = [
            row["methods"][method]["mrr_at_5"]
            for row in all_results
        ]
        latencies = [
            row["methods"][method]["latency_ms"]
            for row in all_results
        ]

        summary[method] = {
            "questions": len(all_results),
            "mean_recall_at_5": round(
                float(np.mean(recalls)), 4
            ),
            "mean_mrr_at_5": round(
                float(np.mean(mrrs)), 4
            ),
            "mean_latency_ms": round(
                float(np.mean(latencies)), 3
            ),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
        }

    output = {
        "top_k": TOP_K,
        "metric_note": (
            "Recall@5 checks whether any top-five chunk comes "
            "from an expected source page. It measures source "
            "retrieval, not whether the exact supporting passage "
            "was retrieved."
        ),
        "summary": summary,
        "results": all_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    with CSV_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "method",
                "recall_at_5",
                "mrr_at_5",
                "latency_ms",
                "top_5_urls",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n=== Overall comparison ===")
    print(
        f"{'Method':<10} {'Recall@5':>10} "
        f"{'MRR@5':>10} {'Mean ms':>10} "
        f"{'P50 ms':>10} {'P95 ms':>10}"
    )

    for method, metrics in summary.items():
        print(
            f"{method:<10} "
            f"{metrics['mean_recall_at_5']:>10.3f} "
            f"{metrics['mean_mrr_at_5']:>10.3f} "
            f"{metrics['mean_latency_ms']:>10.3f} "
            f"{metrics['p50_latency_ms']:>10.3f} "
            f"{metrics['p95_latency_ms']:>10.3f}"
        )

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print(f"CSV results saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()