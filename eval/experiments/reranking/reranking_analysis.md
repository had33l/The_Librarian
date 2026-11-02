# Reranking Experiment — The Librarian

## 1. Objective

The objective was to evaluate whether adding a Cross-Encoder reranking stage to the existing hybrid retrieval pipeline improves the relevance of the top-5 retrieved chunks.

The experiment also measures the additional latency introduced by reranking.

## 2. Experimental setup

- **Evaluation set:** 100 questions.
- **Candidate retrieval:** Hybrid search combining BM25 and dense vector search using Reciprocal Rank Fusion (RRF).
- **Candidate pool:** Top 100 hybrid candidates per question.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Final results:** Top 5 chunks after reranking.
- **Evaluation metrics:** Recall@5, MRR@5 and latency.

Both hybrid retrieval and reranked retrieval were evaluated on the same 100 questions.

## 3. Results

| Metric | Hybrid | Hybrid + reranker |
|---|---:|---:|
| Recall@5 | **95%** | 93% |
| MRR@5 | **0.8858** | 0.8512 |
| Mean latency | 72.71 ms | 2,261.59 ms total |

The reranking stage itself introduced an average latency of approximately **2,188.87 ms** per question.

The candidate pool achieved **99% Recall@100**, meaning that the expected source page was present among the 100 candidates for 99 out of 100 questions.

### Per-question impact

| Outcome | Number of questions |
|---|---:|
| Reranking improved MRR@5 | 7 |
| Reranking worsened MRR@5 | 14 |
| No change | 79 |

Although reranking improved some individual results, it worsened more questions than it improved.

Examples of significant changes:

- Q4: MRR improved from 0.33 to 1.00.
- Q75: MRR improved from 0.00 to 1.00.
- Q3: MRR decreased from 1.00 to 0.00.
- Q46: MRR decreased from 0.50 to 0.00.
- Q66: MRR decreased from 1.00 to 0.00.

## 4. Analysis

The experiment did not demonstrate an improvement from adding this reranking configuration.

There are two main observations:

1. **Retrieval quality decreased:** Recall@5 dropped from 95% to 93%, while MRR@5 decreased from 0.8858 to 0.8512. The reranker sometimes promoted less relevant chunks and pushed useful chunks out of the top 5.
2. **Latency increased substantially:** The Cross-Encoder added approximately 2.19 seconds per question, making the retrieval pipeline considerably slower without improving its overall measured quality.

One possible limitation is the choice of reranker. The selected model, `cross-encoder/ms-marco-MiniLM-L-6-v2`, was trained for passage ranking in a general web-search setting. Its ranking behavior may not be optimal for technical Python documentation. This is a possible explanation, not a proven cause of the results.

The candidate pool also matters: reranking can only reorder retrieved candidates. It cannot recover a relevant passage that was not retrieved in the first place.

## 5. Decision

The Cross-Encoder reranker was **not integrated into the final retrieval pipeline**.

For the current dataset and experimental configuration, hybrid retrieval alone achieved better Recall@5 and MRR@5 while avoiding the additional reranking cost.

This is a negative ablation result: adding the tested reranker did not improve the system. It demonstrates why retrieval components should be evaluated empirically rather than included simply because they are common in RAG architectures.

## 6. Limitations and next steps

- The current relevance evaluation uses the expected source page as its relevance criterion. It does not yet establish whether the exact supporting passage is retrieved. Passage-level evaluation should be added.
- The experiment evaluates one Cross-Encoder model and one candidate-pool configuration. It does not establish that all reranking approaches are ineffective.
- Further optimization should focus on improving hybrid retrieval and measuring end-to-end service latency and cost.

**Conclusion:** Hybrid retrieval remains the selected approach based on the current experimental evidence. The tested reranking stage was rejected because it reduced retrieval quality and introduced substantial latency.