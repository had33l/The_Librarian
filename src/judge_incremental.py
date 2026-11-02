"""LLM-as-judge for a prompt-variant run: incremental, resumable, one retry.

Uses the SAME judge prompt and judge model as src/evaluate_generation_quality.py
(judge_answer is imported, not copied) and the SAME retrieved contexts, so scores
are comparable with the baseline judgements. Differences from the original script:
each judgement is appended to a .jsonl file immediately (an interruption loses
nothing), an interrupted run resumes, and a judge reply that fails JSON parsing
is retried once. Existing output is never overwritten.

  python -m src.judge_incremental \
      --results eval/experiments/prompt_compression/results_lean_all_gpt-5.4-mini.json
"""

import argparse
import json
from pathlib import Path

from src.evaluate_generation_quality import JUDGE_MODEL, judge_answer

CONTEXTS = Path("eval/generation_contexts_openai.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--contexts", type=Path, default=CONTEXTS)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = args.output or args.results.with_name(
        args.results.stem + "_judgements.jsonl")
    results = json.loads(args.results.read_text(encoding="utf-8"))
    contexts = {c["id"]: c for c in json.loads(args.contexts.read_text(encoding="utf-8"))}
    if args.limit:
        results = results[: args.limit]

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip() and "error" not in json.loads(line):
                done.add(json.loads(line)["id"])

    print(f"Judge model: {JUDGE_MODEL} | {len(results)} answers | resumed {len(done)}")
    for i, r in enumerate(results, start=1):
        if r["id"] in done:
            continue
        row = {"id": r["id"], "question": r["question"]}
        for attempt in (1, 2):
            try:
                ev = judge_answer(r, contexts[r["id"]])
                row.update({
                    "faithfulness_score": ev["faithfulness_score"],
                    "correctness_score": ev["correctness_score"],
                    "citation_accuracy_score": ev["citation_accuracy_score"],
                    "unsupported_claims": ev["unsupported_claims"],
                    "citation_issues": ev["citation_issues"],
                    "explanation": ev["explanation"],
                    "attempts": attempt,
                })
                row.pop("error", None)
                break
            except Exception as e:
                row["error"] = str(e)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(results)}] id={r['id']} "
              f"{'ERROR' if 'error' in row else 'ok'}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
