---
title: The Librarian
emoji: 📚
sdk: docker
app_port: 7860
pinned: false
---

# The Librarian

A retrieval-augmented generation (RAG) service over the **Python 3.12 documentation**. Ask a question and get an answer built only from retrieved documentation passages, with `[S1]`-style citations. When the passages do not contain the answer, the system says exactly: *"I don't know based on the available Python documentation."*

Course project, CS496 AI Engineering. Goals: (1) document processing and retrieval, (2) answer generation and evaluation, (3) cutting cost or P95 latency by 50% while losing at most 2 points of accuracy.

## Quick start

**Setup (one command):**

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env
```

Then put your key in `.env` (`OPENAI_API_KEY=...`). `.env` is git-ignored; never commit it or share it in an archive.

**Run (one command):**

```bash
python -m src.ask
```

This starts an interactive prompt that prints the answer, its sources, token usage and timings. The first run needs `data/processed/chunks.json` and `indexes/chunk_embeddings.npy` (see "Build the data and index").

**Web app (backend + frontend, one command):**

```bash
cd backend
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-backend.txt
cp .env.example .env            # then put your OPENAI_API_KEY in it
python -m uvicorn src.api.main:app --port 8000
```

Open http://127.0.0.1:8000 (chat UI), `/docs` (interactive API), `/health` (status). The FastAPI backend serves the `frontend/` UI itself, so there is a single URL. Tests: `cd backend && python -m pytest -v`.

**Deployed service:** **(https://the-librarian-lite1.onrender.com/.)** (free on Render, no card; step-by-step in [`DEPLOY.md`](DEPLOY.md)). Once live, ask a question with:

```bash
curl -X POST <URL>/ask -H "Content-Type: application/json" \
  -d '{"question": "How do I read a file in Python?"}'
```

## How it works

```
question -> BM25 (keywords) --+
                              +-> Reciprocal Rank Fusion -> top 5 chunks -> LLM -> answer + [S#] citations
question -> vector search ----+                                                  \-> or "I don't know"
```

| Component | Choice |
|---|---|
| Corpus | Python 3.12 docs, about 500 pages (`data/processed/documents.json`) |
| Chunking | 800 characters, 120 overlap, recursive splitting, 16,822 chunks |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, stored in `indexes/chunk_embeddings.npy` |
| Retrieval | BM25 + vector search fused with RRF (`RRF_K = 60`), top 5 |
| Reranker | None (tested and rejected) |
| Generation | `gpt-5.4-nano` (cost-optimised, set with `OPENAI_MODEL`) or `gpt-5.4-mini` (default if unset); `reasoning_effort="low"`, max 600 completion tokens |
| Judge | `gpt-5.4-mini` (`JUDGE_MODEL`) |
| Eval sets | `data/eval_questions.json` (100 answerable), `data/unanswerable_questions.json` (20) |

Every design choice, and the known failure cases, are in [`design.md`](design.md).

## Results

### Retrieval ablation (100 questions)

What each retrieval component added:

| Method | Recall@5 | MRR@5 | Added vs previous best |
|---|---:|---:|---|
| BM25 only | 0.910 | 0.7933 | - |
| Vector only | 0.890 | 0.7332 | -0.020 Recall@5 vs BM25 |
| **Hybrid (BM25 + vector, RRF)** | **0.950** | **0.8858** | **+0.040 Recall@5, +0.0925 MRR@5** vs BM25 |
| Hybrid + cross-encoder reranker | 0.930 | 0.8512 | -0.020 Recall@5, -0.0346 MRR@5 vs hybrid, and about +2.19 s per query |

Recall@5 means the expected *source page* is in the top 5, not that the exact supporting passage was retrieved. The reranker candidate pool (top 100) reached Recall@100 = 0.99, and reranking helped 7 questions and hurt 14. Mean retrieval latencies of BM25, vector and hybrid (126, 50 and 162 ms) come from one run; the reranker comparison (hybrid 72.7 ms vs 2,261.6 ms with reranking) comes from another, so compare latencies within a run only. Scripts: `src/evaluate_retrieval.py`, `src/evaluate_reranking.py`.

### Generation, before and after optimisation (100 questions)

Optimisation changed one thing: the generation model, `gpt-5.4-mini` (before) to `gpt-5.4-nano` (after). Retrieval, prompt and context size are unchanged.

| Metric | Before (Mini) | After (Nano) | Change |
|---|---:|---:|---:|
| **Cost per request** | $0.001427 | $0.000373 | **-73.9%** |
| Mean total latency | 1,518 ms | 2,077 ms | +36.8% |
| P50 total latency | 1,455 ms | 1,343 ms | -7.7% |
| **P95 total latency** | 2,093 ms | 2,443 ms | **+16.8%** |
| Faithfulness | 95.96% | 97.47% | +1.52 pp |
| Correctness | 82.83% | 80.81% | -2.02 pp |
| Citation accuracy | 95.96% | 95.45% | -0.51 pp |
| Expected source cited (of 100) | 91 | 90 | -1 |
| Acceptable on 20 unanswerable questions | 20 / 20 | 16 / 20 | -4 |

Quality is compared on 99 paired questions (one Nano judgement failed). Re-running Mini with the same prompt moved scores by 1.5-2.5 pp, so differences of about 2 pp are within noise.

### Does it meet the course target?

| Target from the brief | Result |
|---|---|
| Cut cost per request **or** P95 latency by 50% | **Met through cost (-73.9%)**; P95 is worse (+16.8%) |
| Lose at most 2 points of accuracy | **Narrowly missed on correctness (-2.02 pp)**, within measured noise |
| "I don't know" when it should | Nano is weaker than Mini on false-premise questions |

We report this as a documented trade-off, not a clean win. Full method, tables, the cost model (today and at 100x), and the three rejected experiments are in [`optimisation.md`](optimisation.md). What went wrong and what we learned is in [`postmortem.md`](postmortem.md).

## Brief checklist

| Required by the brief | Where | Status |
|---|---|---|
| README with setup, run, results, Contributions | this file | Contributions: **TODO** |
| Deployed service with a URL | "Deployed service" above | **TODO** |
| `data/`: questions with answers and source passages | `data/eval_questions.json`, `data/unanswerable_questions.json` | Done |
| `eval/`: retrieval eval, generation eval, ablation | `src/evaluate_*.py`, `eval/` | Done (ablation above) |
| `design.md`: chunking, index, failure cases | [`design.md`](design.md) | Done |
| `optimisation.md`: before/after numbers | [`optimisation.md`](optimisation.md) | Done |
| `postmortem.md` | [`postmortem.md`](postmortem.md) | User feedback: **TODO** |
| Cost model | `optimisation.md`, "Cost model" | Done |
| Feedback from real users | `postmortem.md` | **TODO** |
| Report (max 6 pages) | not in the repo | **TODO** |
| Runs from a clean clone, no secrets | - | **TODO**: test before submitting |

## Build the data and index

```bash
python src/inspect_corpus.py        # python-3.12-docs-html/ -> data/processed/documents.json
python src/chunk_corpus.py          # -> data/processed/chunks.json
python -m src.retrieve_vector       # first run builds indexes/chunk_embeddings.npy
python -m src.generate_eval_dataset # (already done) -> data/eval_questions.json
```

## Evaluate

```bash
python -m src.evaluate_retrieval                                   # Recall@5, MRR@5
python -m src.evaluate_reranking                                   # hybrid + reranker ablation
python -m src.evaluate_generation --output eval/<new>.json --csv eval/<new>.csv
python -m src.evaluate_generation --questions data/unanswerable_questions.json --model gpt-5.4-nano --output eval/experiments/abstention_nano.json --csv eval/experiments/abstention_nano.csv
python -m src.evaluate_generation_quality --limit 100 --output eval/<new_judgements>.json
python -m unittest discover -s tests -v                             # latency-statistics unit tests
```

Evaluation scripts refuse to overwrite existing outputs. To reproduce the Mini vs Nano comparison, see the commands at the end of `optimisation.md`.

## Repository layout

```
data/            eval_questions.json, unanswerable_questions.json, processed/{documents,chunks}.json
indexes/         chunk_embeddings.npy                 (rebuilt by src.retrieve_vector; tracked with Git LFS)
src/             research and evaluation code (the experiments). ask.py, chunk_corpus.py, retrieve_{bm25,vector,hybrid}.py,
                 evaluate_{retrieval,generation,generation_quality,reranking}.py,
                 prompt_variants.py, measure_prompt_tokens.py, evaluate_prompt_variant.py,
                 judge_incremental.py, compare_runs.py, latency_metrics.py
tests/           test_latency_metrics.py
backend/         the deployable service: src/api (FastAPI), src/ask.py (RAG pipeline), data/processed,
                 indexes, tests (API tests). Self-contained; this is what the Dockerfile ships.
frontend/        chat UI (index.html, app.js, style.css), served by the backend at /
Dockerfile       full image (PyTorch) for hosts with 2 GB+ RAM (Cloud Run, Hugging Face PRO)
Dockerfile.lite  lightweight image (ONNX) for free 512 MB hosts; used by render.yaml
DEPLOY.md        deployment steps
render.yaml      Render blueprint (free web service)
tools/           count_tokens.js                      (optional token counter; run `npm install` in tools/)
eval/            results, judgements, contexts, experiments/, optimization_methods/
```

## Contributions

**TODO:** one line per team member (max 4) describing what they built or evaluated.

## Ground rules used for experiments

- Never overwrite baseline files; each experiment writes its own outputs.
- The baseline for comparisons is `gpt-5.4-mini` + top-5 hybrid retrieval.
- Same 100 questions, same retrieved chunks and the same judge model for every comparison.
- Compare quality only on questions judged successfully in **both** runs.
- Report mean and P95 latency separately.
- Claim a cost reduction only with real token usage and the applicable model pricing.
