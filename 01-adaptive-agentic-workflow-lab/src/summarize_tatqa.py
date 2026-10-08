"""Summarize one development run: python -m src.summarize_tatqa RESULTS.jsonl."""

import json
import statistics
import sys
from pathlib import Path


def summarize(rows):
    completed = [row for row in rows if "execution_error" not in row]
    return {
        "inputs": len(rows),
        "completed": len(completed),
        "correct": sum(row.get("correct", False) for row in rows),
        "scale_correct": sum(row.get("scale_correct", False) for row in rows),
        "mean_f1": statistics.mean(row["f1"] for row in completed) if completed else None,
        "mean_total_tokens": statistics.mean(row["total_tokens"] for row in completed) if completed else None,
        "median_total_latency_ms": statistics.median(row["latency_ms"] for row in completed) if completed else None,
        "mean_execution_tokens": statistics.mean(row["execution_total_tokens"] for row in completed) if completed else None,
        "median_execution_latency_ms": statistics.median(row["execution_latency_ms"] for row in completed) if completed else None,
        "mean_router_tokens": statistics.mean(row["router_total_tokens"] for row in completed) if completed else None,
        "median_router_latency_ms": statistics.median(row["router_latency_ms"] for row in completed) if completed else None,
    }


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    strategies = list(dict.fromkeys(row["strategy"] for row in rows))
    summary = {
        "source": str(path),
        "scope": "Development only; one run per input; variants share source questions and contexts.",
        "limitations": [
            "20 inputs are variants of 10 questions from five deliberately selected training contexts, not a held-out benchmark.",
            "Strategies ran in fixed order; latency is subject to endpoint and network variation.",
            "Token counts are provider-reported usage, not independently measured compute or monetary cost.",
            "Failed requests have unknown token usage; cost averages include completed requests only, with counts shown.",
            "Exact match may penalize longer text descriptions; raw answers and F1 are retained without manual score overrides.",
            "Expected route labels describe required operations, not empirically optimal routes; direct models may calculate without the added reasoning instruction.",
        ],
        "strategies": {
            strategy: summarize([row for row in rows if row["strategy"] == strategy])
            for strategy in strategies
        },
        "categories": {},
        "routing_errors": [],
        "routing_unavailable": [],
        "answer_failures": [],
    }
    for category in dict.fromkeys(row["id"].split("-", 1)[1] for row in rows):
        summary["categories"][category] = {
            strategy: summarize([
                row for row in rows
                if row["strategy"] == strategy and row["id"].endswith(category)
            ]) for strategy in ["always_complex", "adaptive"]
        }
    adaptive = [row for row in rows if row["strategy"] == "adaptive"]
    for row in adaptive:
        if "execution_error" in row:
            summary["routing_unavailable"].append(row["id"])
        elif any(
            row[key] != value for key, value in row["expected_route"].items()
        ):
            summary["routing_errors"].append(row)
    summary["expected_route_matches"] = len(adaptive) - len(summary["routing_errors"]) - len(summary["routing_unavailable"])
    summary["answer_failures"] = [row for row in rows if not row.get("correct", False)]
    # Describe observable failures, not an assumed internal model thought process.
    for row in summary["answer_failures"]:
        if "execution_error" in row:
            category = "execution_error"
        elif row["error_type"] in ["format_error", "type_error"]:
            category = "invalid_output"
        elif row["expected_route"]["needs_retrieval"] and not row["needs_retrieval"]:
            category = "missing_evidence"
        elif row["needs_retrieval"] and "table" not in row["retrieved_chunk_ids"]:
            category = "inspect_retrieved_evidence"
        elif "-arithmetic-" in row["id"]:
            category = "incorrect_arithmetic_answer_with_evidence"
        else:
            category = "exact_match_mismatch_with_evidence_inspect_semantics"
        row["failure_category"] = category
    baseline = summary["strategies"]["always_complex"]
    adaptive_metrics = summary["strategies"]["adaptive"]
    summary["adaptive_reduction_percent"] = {}
    if baseline["completed"] == baseline["inputs"] and adaptive_metrics["completed"] == adaptive_metrics["inputs"]:
        for metric in ["mean_total_tokens", "median_total_latency_ms", "mean_execution_tokens", "median_execution_latency_ms"]:
            summary["adaptive_reduction_percent"][metric] = (
                (baseline[metric] - adaptive_metrics[metric]) / baseline[metric] * 100
            )
    output = path.with_suffix(".summary.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["strategies"], indent=2))
    print(f"Expected route matches: {summary['expected_route_matches']}/{len(adaptive)}")
    print("Adaptive reductions (negative means an increase):")
    print(json.dumps(summary["adaptive_reduction_percent"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
