"""Run T1, or all 20 development inputs with --all, across five strategies."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.retrieval import TfidfRetriever, load_tatqa_chunks
from src.strategies import run_adaptive, run_fixed
from src.tatqa_scoring import score_tatqa


def main():
    queries = [
        json.loads(line)
        for line in Path("data/tatqa-development.jsonl").read_text().splitlines()
        if line.strip()
    ]
    run_all = "--all" in sys.argv[1:]
    if not run_all:
        queries = [query for query in queries if query["id"].startswith("T1-")]
    contexts = json.loads(Path("data/tatqa-development-contexts.json").read_text())
    retrievers = {
        context["context_id"]: TfidfRetriever(load_tatqa_chunks(context))
        for context in contexts
    }
    strategies = {
        "always_direct": (False, False),
        "always_retrieval": (True, False),
        "always_reasoning": (False, True),
        "always_complex": (True, True),
        "adaptive": None,
    }
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S-%f")
    Path("results").mkdir(exist_ok=True)
    run_name = "development" if run_all else "smoke"
    output_path = Path(f"results/tatqa-{run_name}-{timestamp}.jsonl")
    records = []
    print(f"Saving results to {output_path}", flush=True)

    with output_path.open("w", encoding="utf-8") as output:
        for query in queries:
            retriever = retrievers[query["context_id"]]
            for strategy, flags in strategies.items():
                record = {
                    "id": query["id"],
                    "context_id": query["context_id"],
                    "question": query["question"],
                    "condition": query["condition"],
                    "strategy": strategy,
                    "model": os.getenv("MODEL_NAME"),
                    "gold_answer": query["gold_annotation"]["answer"],
                    "gold_scale": query["gold_annotation"]["scale"],
                    "expected_route": query["expected_route"],
                }
                arguments = dict(
                    question=query["question"],
                    answer_type="tatqa",
                    provided_context=query["provided_context"],
                    retriever=retriever,
                )
                print(f"Running {query['id']} / {strategy}", flush=True)
                try:
                    if strategy == "adaptive":
                        result = run_adaptive(**arguments)
                    else:
                        result = run_fixed(
                            **arguments, needs_retrieval=flags[0], needs_reasoning=flags[1]
                        )
                    record.update(result)
                    # Reconstruct the deterministic retrieval outside the timed execution.
                    record["retrieved_chunk_ids"] = [
                        chunk.chunk_id for chunk in retriever.retrieve(query["question"])
                    ] if result["needs_retrieval"] else []
                    record.update(score_tatqa(result["answer_text"], query["gold_annotation"]))
                except Exception as error:
                    record["execution_error"] = f"{type(error).__name__}: {error}"
                records.append(record)
                line = json.dumps(record, ensure_ascii=False)
                output.write(line + "\n")
                output.flush()
                print(line, flush=True)

    print("\nDevelopment summary (one run per input):", flush=True)
    for strategy in strategies:
        rows = [record for record in records if record["strategy"] == strategy]
        correct = sum(record.get("correct", False) for record in rows)
        errors = sum("execution_error" in record for record in rows)
        print(f"{strategy}: {correct}/{len(rows)} correct; {errors} execution errors")


if __name__ == "__main__":
    main()
