"""Offline prompt-token measurement for the input-token-reduction experiment.

No API calls. Builds the exact prompt each variant would send for the 100
evaluation questions (using the saved retrieved contexts) and counts tokens with
the o200k_base encoding. Validation: for the baseline prompt this count plus a
constant 11-token chat-framing overhead equals the API-reported input_tokens of
the nano run on all 100 questions (see --validate).

Counter: tiktoken if it can load o200k_base, otherwise Node + gpt-tokenizer
(tools/count_tokens.js; `npm install gpt-tokenizer` next to it).

Usage:
  python -m src.measure_prompt_tokens
  python -m src.measure_prompt_tokens --validate eval/experiments/nano_generation_results.json
"""

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

from src.prompt_variants import VARIANTS, build_messages, short_title

CONTEXTS = Path("eval/generation_contexts_openai.json")
OUT_DIR = Path("eval/experiments/prompt_compression")
CHAT_OVERHEAD = 11  # constant framing tokens measured against real API usage


def make_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return "tiktoken", lambda texts: [len(enc.encode(t)) for t in texts]
    except Exception:
        def node_counter(texts):
            out = subprocess.run(
                ["node", "tools/count_tokens.js"],
                input=json.dumps(texts), capture_output=True, text=True,
                check=True,
            ).stdout
            return json.loads(out)
        return "node gpt-tokenizer", node_counter


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, default=CONTEXTS)
    ap.add_argument("--validate", type=Path, default=None,
                    help="Results file with real input_tokens (baseline prompt).")
    args = ap.parse_args()

    contexts = json.loads(args.contexts.read_text(encoding="utf-8"))
    counter_name, count = make_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    totals = {}
    for variant in VARIANTS:
        prompts = []
        for item in contexts:
            msgs = build_messages(item["question"], item["retrieved_chunks"], variant)
            prompts.append("\n".join(m["content"] for m in msgs))
        totals[variant] = [n + CHAT_OVERHEAD for n in count(prompts)]

    # Component breakdown of the baseline prompt.
    comp_texts = {"system_prompt": [], "question": [], "passage_text": [],
                  "title_field": [], "url_field": [], "labels_and_format": []}
    for item in contexts:
        msgs = build_messages(item["question"], item["retrieved_chunks"], "baseline")
        chunks = item["retrieved_chunks"]
        comp_texts["system_prompt"].append(msgs[0]["content"])
        comp_texts["question"].append(item["question"])
        comp_texts["passage_text"].append("\n".join(c["text"] for c in chunks))
        comp_texts["title_field"].append("\n".join(c["title"] for c in chunks))
        comp_texts["url_field"].append("\n".join(c["url"] for c in chunks))
    comp = {k: count(v) for k, v in comp_texts.items() if v}
    base = totals["baseline"]
    comp_mean = {k: st.mean(v) for k, v in comp.items()}
    comp_mean["labels_and_format"] = max(
        0.0, st.mean(base) - sum(comp_mean.values()))

    summary = {
        "counter": counter_name,
        "questions": len(contexts),
        "chat_overhead_tokens": CHAT_OVERHEAD,
        "baseline_component_mean_tokens": {k: round(v, 1) for k, v in comp_mean.items()},
        "variants": {},
    }
    for variant, vals in totals.items():
        summary["variants"][variant] = {
            "mean_input_tokens": round(st.mean(vals), 1),
            "min": min(vals), "max": max(vals),
            "total_100q": sum(vals),
            "change_vs_baseline_pct": round(pct(st.mean(vals), st.mean(base)), 2),
        }

    if args.validate:
        real = {r["id"]: r["input_tokens"]
                for r in json.loads(args.validate.read_text(encoding="utf-8"))
                if not r.get("error")}
        diffs = [real[c["id"]] - totals["baseline"][i]
                 for i, c in enumerate(contexts) if c["id"] in real]
        summary["validation"] = {
            "file": str(args.validate), "questions": len(diffs),
            "real_minus_counted_min": min(diffs), "max": max(diffs),
            "exact_match_for_all": min(diffs) == max(diffs) == 0,
        }

    (OUT_DIR / "token_measurement.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Counter: {counter_name}")
    print("\nBaseline prompt, mean tokens per question:")
    for k, v in summary["baseline_component_mean_tokens"].items():
        print(f"  {k:<18}{v:>8}")
    print(f"  {'TOTAL':<18}{round(st.mean(base), 1):>8}")
    print("\nVariant                        mean   change   (100 questions)")
    for v, s in summary["variants"].items():
        print(f"  {v:<30}{s['mean_input_tokens']:>7}  {s['change_vs_baseline_pct']:>6}%  {s['total_100q']:>8}")
    if args.validate:
        print("\nValidation:", summary["validation"])


if __name__ == "__main__":
    main()
