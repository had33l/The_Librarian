# Experiment 4 - cost reduction: lean prompt + nano (runbook)

Course target: cut **cost per request OR P95 latency by 50%**, losing at most **2 pp** of quality, measured before/after on the same 100 questions.

## What the numbers say before spending anything

Cost per request = input_tokens x input price + output_tokens x output price. List prices used (third-party price trackers, all agreeing; **confirm on OpenAI's pricing page**): `gpt-5.4-mini` $0.75 in / $4.50 out per 1M; `gpt-5.4-nano` $0.20 in / $1.25 out per 1M.

| Configuration | Input tok | Output tok | Cost / request | vs mini baseline |
|---|---:|---:|---:|---:|
| mini + baseline prompt | 1,191 (measured) | ~105 (assumed, not measured) | ~$0.00137 | - |
| mini + `lean_all` prompt | 978.5 (measured offline) | ~105 | ~$0.00121 | ~ -12% |
| nano + baseline prompt | 1,191 (measured) | 107.6 (measured) | ~$0.00037 | ~ -73% |
| nano + `lean_all` prompt | 978.5 | 107.6 | ~$0.00033 | ~ -76% |

Trimming the prompt alone does **not** reach 50%. The model switch does; the lean prompt adds a few points on top. Mini output tokens were never saved, so the mini rows are estimates (the result stays between -70% and -79% for nano if mini outputs were anywhere from 80 to 150 tokens). Trade-off to report honestly: nano was slower (mean total 2,077 ms vs 1,570 ms; P95 about 2,443 ms vs 2,170 ms), so the target is met on **cost**, not on P95 latency.

## Step A - zero API cost: nano vs mini on the data you already have

The original mini run never saved tokens, but the nano run used the IDENTICAL prompt and recorded real input tokens, and the offline count matches real API usage exactly (also confirmed on mini: pilot input tokens 980.6 = offline count). Mini's output tokens were never recorded, so the script assumes 0 for them, which makes mini look as cheap as possible: the reported cost reduction is a **floor**.

```bash
python -m src.compare_runs \
  --base-results eval/generation_results_openai.json \
  --base-judgements eval/generation_quality_judgements.json \
  --new-results eval/experiments/nano_generation_results.json \
  --new-judgements eval/experiments/nano_generation_quality_judgements.json \
  --base-usage-from eval/experiments/nano_generation_results.json \
  --input-price-per-1m 0.75 --output-price-per-1m 4.50 \
  --new-input-price-per-1m 0.20 --new-output-price-per-1m 1.25
```

Result on the 98 paired questions: faithfulness -0.51 pp, correctness -0.51 pp, citation accuracy +1.02 pp; cost per request at least **58% lower** ($0.000893 -> $0.000373, floor); mean total latency +32%, P95 +12.5% (target met on cost, not on P95).

## Step B - optional, about $0.23: add the lean prompt on top of nano

```bash
python -m src.evaluate_prompt_variant --variant lean_all --model gpt-5.4-nano
python -m src.judge_incremental --results eval/experiments/prompt_compression/results_lean_all_gpt-5.4-nano.json
python -m src.compare_runs --base-results eval/generation_results_openai.json \
  --base-judgements eval/generation_quality_judgements.json \
  --new-results eval/experiments/prompt_compression/results_lean_all_gpt-5.4-nano.json \
  --new-judgements eval/experiments/prompt_compression/results_lean_all_gpt-5.4-nano_judgements.jsonl \
  --base-usage-from eval/experiments/nano_generation_results.json \
  --input-price-per-1m 0.75 --output-price-per-1m 4.50 \
  --new-input-price-per-1m 0.20 --new-output-price-per-1m 1.25
```
Expected extra saving is small (about 18% fewer input tokens, a few more points of cost). Only adopt it if quality stays within 2 pp.

## Other variants (optional, only if budget remains)

`lean_context`, `lean_context_compact_system` and the mini + `lean_all` combination are all runnable the same way with `--variant` / `--model`. Offline token counts are in `token_measurement.json`.

## Notes
- Every run writes a `.jsonl` after each question and resumes if re-run; final files are never overwritten.
- Prompt caching does not apply here as far as I know: the fixed prefix (system prompt, ~167 tokens) is far below the minimum cacheable prefix (1,024 tokens on OpenAI; verify in the docs).
- A cache for repeated questions cannot be evaluated on this test set (no repeats), so it should be described as an option, not claimed as a result.
