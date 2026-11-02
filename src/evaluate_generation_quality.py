import json
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

RESULTS_FILE = Path("eval/generation_results_openai.json")
CONTEXTS_FILE = Path("eval/generation_contexts_openai.json")
OUTPUT_FILE = Path("eval/generation_quality_judgements.json")

JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
)


def judge_answer(result, context):
    passages = []

    for i, chunk in enumerate(context["retrieved_chunks"], start=1):
        passages.append(
            f"[S{i}]\n"
            f"Title: {chunk['title']}\n"
            f"URL: {chunk['url']}\n"
            f"Passage: {chunk['text']}"
        )

    prompt = f"""
Evaluate this RAG-generated answer.

QUESTION:
{result['question']}

EXPECTED ANSWER:
{result['expected_answer']}

SUPPORTING PASSAGE:
{result['supporting_passage']}

RETRIEVED DOCUMENTATION:
{"".join(chr(10) + chr(10) + p for p in passages)}

GENERATED ANSWER:
{result['answer']}

Evaluate three dimensions independently:

1. faithfulness:
Are the factual claims in the answer supported by the retrieved
documentation? Do not reward claims based on outside knowledge.

2. correctness:
Does the answer correctly and sufficiently address the question?
Use the expected answer and supporting passage as references.
Accept equivalent wording and technically correct explanations.

3. citation_accuracy:
Do the citations [S1], [S2], etc. actually support the claims
they are attached to? Penalize missing citations for factual claims
and citations that do not support their associated claims.

For each dimension, assign:
- 1.0: Fully satisfactory
- 0.5: Partially satisfactory
- 0.0: Unsatisfactory

Return ONLY valid JSON with this structure:
{{
  "faithfulness_score": 0.0,
  "correctness_score": 0.0,
  "citation_accuracy_score": 0.0,
  "unsupported_claims": [],
  "citation_issues": [],
  "explanation": "Brief explanation"
}}
"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of retrieval-augmented "
                    "generation systems. Evaluate only the provided "
                    "evidence. Be objective and do not assume claims "
                    "are correct merely because they sound plausible."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_completion_tokens=700,
    )

    evaluation = json.loads(
        response.choices[0].message.content
    )

    return evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval/generation_results_openai.json"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("eval/generation_contexts_openai.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/generation_quality_judgements.json"),
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists. Choose another output path."
        )

    results = json.loads(args.results.read_text(encoding="utf-8"))
    contexts = json.loads(args.contexts.read_text(encoding="utf-8"))

    if len(results) != len(contexts):
        raise ValueError("Results and contexts counts differ.")

    context_by_id = {item["id"]: item for item in contexts}
    selected = results[:args.limit]
    evaluations = []

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Evaluating {len(selected)} answers...")

    for index, result in enumerate(selected, start=1):
        context = context_by_id.get(result["id"])

        if context is None:
            raise ValueError(f"Missing context for question {result['id']}")

        print(f"Evaluating {index}/{len(selected)}...")

        try:
            evaluation = judge_answer(result, context)

            evaluations.append({
                "id": result["id"],
                "question": result["question"],
                "faithfulness_score": evaluation["faithfulness_score"],
                "correctness_score": evaluation["correctness_score"],
                "citation_accuracy_score": evaluation["citation_accuracy_score"],
                "unsupported_claims": evaluation["unsupported_claims"],
                "citation_issues": evaluation["citation_issues"],
                "explanation": evaluation["explanation"],
            })

        except Exception as error:
            print(f"Error on question {result['id']}: {error}")
            evaluations.append({
                "id": result["id"],
                "question": result["question"],
                "error": str(error),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    successful = [e for e in evaluations if "error" not in e]

    print("\nEvaluation complete.")
    print(f"Successful: {len(successful)}/{len(selected)}")
    print(f"Saved to: {args.output}")

    if successful:
        for metric in (
            "faithfulness_score",
            "correctness_score",
            "citation_accuracy_score",
        ):
            average = sum(e[metric] for e in successful) / len(successful)
            print(f"Average {metric}: {average:.2%}")


if __name__ == "__main__":
    main()