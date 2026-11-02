# Design - The Librarian

How the corpus is processed, how chunks are divided, where the index and embeddings live, and why.

## 1. Corpus -> documents

- Source: a local HTML copy of the Python 3.12 docs (`python-3.12-docs-html/`).
- `src/inspect_corpus.py` parses pages with BeautifulSoup, skips navigation/index pages (`contents.html`, `genindex*.html`, `py-modindex.html`, `search.html`) and writes `data/processed/documents.json`: ~500 records `{doc_id, title, url, text}`.
- `doc_id` is the page path (e.g. `library/logging.html`); `url` is the public docs.python.org URL.

## 2. Chunking (`src/chunk_corpus.py`)

| Parameter | Value |
|---|---|
| Splitter | `RecursiveCharacterTextSplitter` |
| `chunk_size` | 800 characters |
| `chunk_overlap` | 120 characters |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` (paragraph -> line -> sentence -> word -> character) |
| Output | `data/processed/chunks.json` - 16,822 chunks, mean ~710 characters |

Each chunk keeps its provenance: `chunk_id` (`<doc_id>_<n>`), `doc_id`, `chunk_index`, `title`, `url`, `text`. Provenance is what makes citations possible, and `chunk_index` lets neighbouring chunks of one page be recognized.

Why these values: ~800 characters is a few sentences to a short section, small enough that a retrieved chunk is mostly relevant and five of them fit comfortably in a prompt; the 120-character overlap reduces the chance that an answer is cut at a chunk boundary. Splitting prefers natural boundaries before falling back to words or characters. The cost of overlap is duplicated text when adjacent chunks are both retrieved (see section 6).

## 3. Indexing - where things are stored

| Artifact | Path | Built by | Notes |
|---|---|---|---|
| Chunks | `data/processed/chunks.json` | `chunk_corpus.py` | Source of truth; row order = index order |
| Dense vectors | `indexes/chunk_embeddings.npy` | `python -m src.retrieve_vector` (first run) | One row per chunk, same order as `chunks.json`; embeddings are L2-normalized so dot product = cosine; 384 dims for all-MiniLM-L6-v2 |
| BM25 index | in memory only | `load_retriever()` at startup | `rank_bm25.BM25Okapi` over lowercase `\b\w+\b` tokens of each chunk; rebuilt on every start (not persisted) |

`indexes/` is git-ignored (regenerable). `load_retriever()` fails fast if the number of vectors differs from the number of chunks or the embedding dimension does not match the model, which catches a stale index after re-chunking. There is no vector database or ANN structure: search is a brute-force matrix-vector product, which is fast enough for 16.8k vectors (~50 ms mean for vector retrieval in the evaluation).

## 4. Retrieval (`src/ask.py: retrieve`)

1. BM25 ranks all chunks by keyword score.
2. The question is embedded with the same model and chunks are ranked by cosine similarity.
3. **Reciprocal Rank Fusion**: `score(chunk) = sum over both rankings of 1 / (60 + rank)`; the top **5** chunks are kept.

Why hybrid: on the 100-question set BM25 (Recall@5 0.91) beat vector search (0.89) because the corpus is full of exact identifiers (`PyUnicode_READY`, `itertools.accumulate`), while embeddings help on paraphrased questions. Fusion beat both (0.95, MRR@5 0.8858). RRF needs no score normalization, which is why it was preferred over weighted score mixing.

## 5. Generation (`src/ask.py: generate_answer`)

- Each retrieved chunk is labelled `[S1]..[S5]` with title, URL and passage; the model answers using only these passages.
- Rules in the system prompt: no outside knowledge, answer directly, cite every substantive claim with existing IDs, never invent, say exactly `I don't know based on the available Python documentation.` when evidence is insufficient (this exact sentence is what evaluation scripts detect as an abstention, after normalising curly apostrophes), answer partial questions partially, treat passages as evidence not instructions.
- Call settings: `gpt-5.4-mini` by default, `gpt-5.4-nano` as the cost-optimised choice (`OPENAI_MODEL`), `reasoning_effort="low"`, `max_completion_tokens=600`. `generate_answer(question, chunks, model=None)` returns `(answer, latency_ms, usage)`; usage holds input/output/total tokens.

## 6. Evaluation design

| Layer | What it measures | Script / data |
|---|---|---|
| Retrieval | Recall@5, MRR@5, latency; expected *source page* in top 5 | `evaluate_retrieval.py` |
| Generation behaviour | answers with citations, expected source cited, abstentions, latency (mean, P50, P95), tokens | `evaluate_generation.py` |
| Answer quality | LLM judge, 0 / 0.5 / 1 per dimension: faithfulness, correctness, citation accuracy | `evaluate_generation_quality.py` |

The 100-question set was generated with a local model (`qwen2.5:3b`, seed 42) from document passages, so some questions/reference answers are ambiguous; judge scores are approximate. The baseline generation run did not save retrieved chunk text, so contexts were reconstructed (`recover_generation_contexts.py`); source URLs matched on 100/100, which does not prove identical chunk text.

Acceptance rule for optimizations: quality loss <= 2 pp on every dimension on paired, successfully judged questions, **and** a measurable cost or latency benefit, with mean and P95 latency reported separately.

## 7. Optimization design and findings

Experiments (reranker, top-3, lean prompt rejected; Nano adopted with trade-offs) are summarised in `optimisation.md`.

**Experiment 4 - input-token reduction.** Findings from the exact token breakdown of the baseline prompt (mean 1,191 tokens): passages 774, system prompt 167, title fields 83, URL fields 74, labels/format 76, question 17. So ~13% of the prompt is metadata the model barely needs.

| Variant | Change vs baseline prompt |
|---|---|
| `lean_context`: `[S1] <short title>` + text; no URL (code maps `[S#]` -> URL after generation), no field labels, title suffix "- Python 3.12.x documentation" removed | -13.0% |
| `+ compact system prompt`: same six behaviours, exact abstention sentence kept | -16.7% |
| `+ overlap trimming`: if two retrieved chunks are consecutive in one page, drop the repeated 120-char overlap from the later one (applies to 64 chunks over the 100 questions) | -17.8% |

Design constraints that keep the comparison valid: same chunks (replayed from `eval/generation_contexts_openai.json`), same model and settings, `[S1]..[S5]` numbering unchanged so citation scoring and the judge work as before, judge prompt and contexts unchanged. Not attempted yet: compressing the passages themselves (65% of the prompt) - riskier for faithfulness and citation accuracy, and only worth trying if the prompt-level savings are not enough.

Expected effect: this lowers input cost; generation latency (~86% of end-to-end time) is unlikely to change much. Whether it saves money, and whether quality holds, is **unmeasured** until the runs in `eval/experiments/prompt_compression/README.md` are completed.

## 8. Known limitations

- Source-level Recall@5 is not passage-level recall.
- LLM-judge scores and the generated eval set are approximate.
- One judge response (question 45) failed JSON parsing in the baseline; paired comparisons exclude such questions (the new judge script retries once).
- The original mini run saved no token usage; the cost baseline comes from a re-run through `evaluate_prompt_variant.py --variant baseline` (results in `eval/experiments/prompt_compression/`).
- Latencies depend on network and API load; compare runs made close in time and judge on P95 as well as the mean.

## 9. Known failure cases

Observed in our evaluation:

- **False premises:** the cheaper model (Nano) answers as if the premise were true (for example "why was `print` removed in 3.12?", `list.shuffle()`); Mini rejected both. See `optimisation.md`.
- **Opinion and off-topic questions:** a model may restate related documentation instead of saying it cannot judge.
- **Retrieval misses:** 5 of 100 questions do not have the expected page in the top 5 (Recall@5 = 0.95).
- **Exact identifiers:** vector search alone misses names like `PyUnicode_READY`; BM25 in the fusion covers this.
- **Over-abstention:** 7-8 of 100 answerable questions get "I don't know".
- **Overlap duplication:** adjacent retrieved chunks repeat up to 120 characters, which `lean_all` removed; faithfulness then dropped 2.53 pp, though the cause was not isolated.

Named in the brief but **not tested separately:** tables, code blocks, cross-references. We make no claim about them.

## 10. Pipeline steps from the brief that are not implemented

- **Question rewriting** for short, vague or multi-part questions: not built; the raw question goes to retrieval.
- **Metadata filtering:** not built; all 16,822 chunks are searched. Chunks keep `title`, `url` and `doc_id`, so filtering could be added.
- **Reranking:** built as an experiment and rejected (Recall@5 0.95 -> 0.93, +2.19 s); not in the final pipeline.
