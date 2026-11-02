import csv
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


CHUNKS_FILE = Path("data/processed/chunks.json")
QUESTIONS_FILE = Path("data/eval_questions.json")
INDEX_FILE = Path("indexes/chunk_embeddings.npy")

OUTPUT_DIR = Path("eval/experiments/reranking")
RESULTS_FILE = OUTPUT_DIR / "reranking_results.json"
CSV_FILE = OUTPUT_DIR / "reranking_results.csv"

RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 100
TOP_K = 5
RRF_K = 60
RERANK_BATCH_SIZE = 32


def tokenize(text):
    return __import__("re").findall(r"\b\w+\b", text.lower())


def top_candidate_ranking(scores, candidate_k=CANDIDATE_K):
    candidate_k = min(candidate_k, len(scores))
    candidates = np.argpartition(scores, -candidate_k)[-candidate_k:]
    return candidates[np.argsort(scores[candidates])[::-1]]


def get_source(chunk):
    return chunk["url"].rstrip("/")


def normalize_sources(sources):
    return {source.rstrip("/") for source in sources}


def reciprocal_rank(expected, ranking, chunks):
    for rank, index in enumerate(ranking, start=1):
        if get_source(chunks[int(index)]) in expected:
            return 1.0 / rank
    return 0.0


def retrieve_hybrid(question, bm25, model, embeddings):
    """Reproduce the existing BM25 + vector + RRF retrieval."""
    start = time.perf_counter()

    keyword_scores = bm25.get_scores(tokenize(question))

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    vector_scores = embeddings @ query_vector

    keyword_ranking = top_candidate_ranking(keyword_scores)
    vector_ranking = top_candidate_ranking(vector_scores)

    rrf_scores = {}

    for candidate_ranking in (keyword_ranking, vector_ranking):
        for rank, index in enumerate(candidate_ranking, start=1):
            index = int(index)
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    # Same RRF ranking as the existing evaluator.
    full_ranking = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    hybrid_latency = (time.perf_counter() - start) * 1000

    # Reranker receives the first 100 candidates from the fused ranking.
    candidates = full_ranking[:CANDIDATE_K]

    return full_ranking, candidates, hybrid_latency


def main():
    if RESULTS_FILE.exists() or CSV_FILE.exists():
        raise FileExistsError(
            "Reranking output already exists. "
            "Rename or back it up before running again."
        )

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = json.load(f)

    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        questions = json.load(f)

    embeddings = np.load(INDEX_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding index and chunks do not match.")

    print("Preparing BM25...")
    bm25 = BM25Okapi([
        tokenize(chunk["text"]) for chunk in chunks
    ])

    print("Loading retrieval model...")
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL)

    print("Loading Cross-Encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    # Warm up both models. Loading and warm-up are excluded from latency.
    retrieval_model.encode(["warm up"], convert_to_numpy=True)
    reranker.predict(
        [("warm up question", "warm up passage")],
        show_progress_bar=False,
    )

    all_results = []
    csv_rows = []

    print(f"\nEvaluating {len(questions)} questions...\n")

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        expected_sources = item.get(
            "expected_sources",
            item.get("expected_source"),
        )

        if isinstance(expected_sources, str):
            expected_sources = [expected_sources]

        if not expected_sources:
            raise ValueError(f"Question {number} has no expected source.")

        expected = normalize_sources(expected_sources)

        # Step 1: Hybrid retrieval.
        full_ranking, candidates, hybrid_latency = retrieve_hybrid(
            question, bm25, retrieval_model, embeddings
        )

        # Existing hybrid top-5 baseline.
        hybrid_top5 = full_ranking[:TOP_K]

        # Step 2: Measure whether the expected source is in the candidate pool.
        candidate_recall = float(any(
            get_source(chunks[i]) in expected for i in candidates
        ))

        # Step 3: Score each question-passage pair with the Cross-Encoder.
        pairs = [
            (question, chunks[index]["text"])
            for index in candidates
        ]

        start = time.perf_counter()
        scores = reranker.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        reranking_latency = (time.perf_counter() - start) * 1000

        reranked_order = np.argsort(scores)[::-1]
        reranked_top5 = [
            candidates[int(position)]
            for position in reranked_order[:TOP_K]
        ]

        # Step 4: Calculate metrics.
        hybrid_recall = float(any(
            get_source(chunks[i]) in expected for i in hybrid_top5
        ))
        reranked_recall = float(any(
            get_source(chunks[i]) in expected for i in reranked_top5
        ))

        hybrid_mrr = reciprocal_rank(expected, hybrid_top5, chunks)
        reranked_mrr = reciprocal_rank(expected, reranked_top5, chunks)

        total_latency = hybrid_latency + reranking_latency

        def describe(indices):
            return [
                {
                    "title": chunks[i]["title"],
                    "url": chunks[i]["url"],
                    "chunk_id": chunks[i]["chunk_id"],
                }
                for i in indices
            ]

        record = {
            "question": question,
            "expected_sources": expected_sources,
            "candidate_recall_at_100": candidate_recall,
            "hybrid": {
                "recall_at_5": hybrid_recall,
                "mrr_at_5": hybrid_mrr,
                "latency_ms": round(hybrid_latency, 3),
                "top_results": describe(hybrid_top5),
            },
            "reranked": {
                "recall_at_5": reranked_recall,
                "mrr_at_5": reranked_mrr,
                "reranking_latency_ms": round(reranking_latency, 3),
                "total_latency_ms": round(total_latency, 3),
                "top_results": describe(reranked_top5),
            },
        }

        all_results.append(record)

        csv_rows.append({
            "question": question,
            "candidate_recall_at_100": candidate_recall,
            "hybrid_recall_at_5": hybrid_recall,
            "hybrid_mrr_at_5": hybrid_mrr,
            "hybrid_latency_ms": round(hybrid_latency, 3),
            "reranked_recall_at_5": reranked_recall,
            "reranked_mrr_at_5": reranked_mrr,
            "reranking_latency_ms": round(reranking_latency, 3),
            "total_latency_ms": round(total_latency, 3),
            "hybrid_top_5_urls": " | ".join(
                chunks[i]["url"] for i in hybrid_top5
            ),
            "reranked_top_5_urls": " | ".join(
                chunks[i]["url"] for i in reranked_top5
            ),
        })

        print(
            f"{number:3}/{len(questions)} | "
            f"Candidate R@100={candidate_recall:.0f} | "
            f"Hybrid R@5={hybrid_recall:.0f}, "
            f"MRR={hybrid_mrr:.3f} | "
            f"Reranked R@5={reranked_recall:.0f}, "
            f"MRR={reranked_mrr:.3f} | "
            f"Total={total_latency:.1f} ms"
        )

    # Aggregate metrics across all questions.
    def mean(path):
        values = [
            row[path[0]][path[1]]
            if len(path) == 2 else row[path[0]]
            for row in all_results
        ]
        return round(float(np.mean(values)), 4)

    summary = {
        "questions": len(all_results),
        "candidate_recall_at_100": round(
            float(np.mean([
                row["candidate_recall_at_100"]
                for row in all_results
            ])), 4
        ),
        "hybrid": {
            "mean_recall_at_5": mean(("hybrid", "recall_at_5")),
            "mean_mrr_at_5": mean(("hybrid", "mrr_at_5")),
            "mean_latency_ms": mean(("hybrid", "latency_ms")),
        },
        "reranked": {
            "mean_recall_at_5": mean(("reranked", "recall_at_5")),
            "mean_mrr_at_5": mean(("reranked", "mrr_at_5")),
            "mean_reranking_latency_ms": mean(
                ("reranked", "reranking_latency_ms")
            ),
            "mean_total_latency_ms": mean(
                ("reranked", "total_latency_ms")
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump({
            "models": {
                "retrieval": RETRIEVAL_MODEL,
                "reranker": RERANKER_MODEL,
            },
            "configuration": {
                "candidate_k": CANDIDATE_K,
                "top_k": TOP_K,
                "rrf_k": RRF_K,
            },
            "summary": summary,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved JSON: {RESULTS_FILE}")
    print(f"Saved CSV:  {CSV_FILE}")


if __name__ == "__main__":
    main()
