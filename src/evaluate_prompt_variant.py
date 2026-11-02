"""Run generation for one prompt variant on the 100 evaluation questions.

Controlled comparison: the SAME retrieved chunks the baseline used (replayed from
eval/generation_contexts_openai.json) are formatted by the chosen variant and sent
to the SAME model with the SAME settings (reasoning_effort="low", 600 max tokens).
Only the prompt changes. Retrieval is therefore not re-run; retrieval_ms is copied
from the baseline results file for the same question id (retrieval is unchanged).

Run the baseline variant too, so the cost baseline has real token usage:
  python -m src.evaluate_prompt_variant --variant baseline
  python -m src.evaluate_prompt_variant --variant lean_all

Results are appended to a .jsonl file after every question; an interrupted run is
resumed by re-running the same command. Existing result files are never overwritten.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.prompt_variants import VARIANTS, build_messages

load_dotenv()

QUESTIONS_FILE = Path("data/eval_questions.json")
CONTEXTS_FILE = Path("eval/generation_contexts_openai.json")
BASELINE_RESULTS = Path("eval/generation_results_openai.json")
OUT_DIR = Path("eval/experiments/prompt_compression")
ABSTENTION = "I don't know based on the available Python documentation."
MAX_NEW_TOKENS = 600


def expected_sources(item):
    s = item.get("expected_sources", item.get("expected_source", []))
    return [s] if isinstance(s, str) else s


def citation_metrics(answer, chunks, expected):
    cited = set(re.findall(r"\[S(\d+)\]", answer))
    valid = {str(i) for i in range(1, len(chunks) + 1)}
    ok, bad = cited & valid, cited - valid
    urls = [chunks[int(i) - 1]["url"] for i in ok]
    return {
        "citation_count": len(cited),
        "valid_citation_count": len(ok),
        "invalid_citations": sorted(bad),
        "expected_source_cited": any(u in urls for u in expected),
        "cited_urls": urls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    client = OpenAI()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{args.variant}_{args.model}"
    jsonl = args.out_dir / f"{stem}.jsonl"
    final_json = args.out_dir / f"{stem}.json"
    final_csv = args.out_dir / f"{stem}.csv"
    if final_json.exists() or final_csv.exists():
        raise FileExistsError(f"{final_json} already exists; choose another --out-dir.")

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    contexts = {c["id"]: c for c in json.loads(CONTEXTS_FILE.read_text(encoding="utf-8"))}
    base = {r["id"]: r for r in json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))}
    if args.limit:
        questions = questions[: args.limit]

    done = {}
    if jsonl.exists():  # resume
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done[r["id"]] = r

    print(f"Variant={args.variant} model={args.model} resumed={len(done)}")
    for index, item in enumerate(questions, start=1):
        qid = list(base)[index - 1]  # baseline ids follow question file order
        if qid in done:
            continue
        ctx = contexts[qid]
        chunks = ctx["retrieved_chunks"]
        exp = expected_sources(item)
        print(f"[{index}/{len(questions)}] {item['question']}")
        try:
            messages = build_messages(item["question"], chunks, args.variant)
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                reasoning_effort="low",
                max_completion_tokens=MAX_NEW_TOKENS,
            )
            gen_ms = (time.perf_counter() - start) * 1000
            answer = (resp.choices[0].message.content or "").strip()
            retrieval_ms = base[qid]["retrieval_ms"]
            row = {
                "id": qid,
                "variant": args.variant,
                "model": args.model,
                "question": item["question"],
                "expected_answer": item.get("expected_answer", ""),
                "supporting_passage": item.get("supporting_passage", ""),
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
                "expected_sources": exp,
                "answer": answer,
                "retrieved_sources": [c["url"] for c in chunks],
                **citation_metrics(answer, chunks, exp),
                "abstained": ABSTENTION.lower() in answer.replace("\u2019", "'").lower(),
                "retrieval_ms": retrieval_ms,
                "generation_ms": round(gen_ms, 2),
                "total_ms": round(retrieval_ms + gen_ms, 2),
                "error": None,
            }
        except Exception as error:  # keep going; failed ids are retried on resume
            row = {"id": qid, "variant": args.variant, "question": item["question"],
                   "error": str(error)}
        with jsonl.open("a", encoding="utf-8") as f:  # incremental save
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row.get("error"):
            done[qid] = row
        print("   ", (row.get("answer") or row.get("error"))[:140].replace("\n", " "))

    rows = [done[k] for k in sorted(done)]
    final_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with final_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        print(f"\n{n} answers | mean input tokens "
              f"{sum(r['input_tokens'] for r in rows)/n:.1f} | mean output "
              f"{sum(r['output_tokens'] for r in rows)/n:.1f} | mean generation "
              f"{sum(r['generation_ms'] for r in rows)/n:.1f} ms")
    print(f"Saved {final_json}")


if __name__ == "__main__":
    main()
