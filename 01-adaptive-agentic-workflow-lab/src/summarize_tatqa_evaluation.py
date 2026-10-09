"""Offline summary of a complete frozen run; never filter costs by correctness."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


def average(rows, field, median=False):
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return (statistics.median(values) if median else statistics.mean(values)) if values else None


def describe(rows):
    inputs = defaultdict(list)
    for row in rows:
        inputs[row["id"]].append(row)
    return {
        "executions": len(rows),
        "inputs": len(inputs),
        "majority_correct_inputs": sum(sum(r["correct"] for r in group) >= 2 for group in inputs.values()),
        "correct_executions": sum(r["correct"] for r in rows),
        "mean_f1_failure_inclusive": statistics.mean(r["f1"] for r in rows),
        "literal_scale_correct_executions": sum(r.get("scale_correct", False) for r in rows),
        "errors": dict(Counter(r["error_type"] or "none" for r in rows)),
        "unknown_usage_executions": sum(r.get("total_tokens") is None for r in rows),
        "observed_latency_executions": sum(r.get("latency_ms") is not None for r in rows),
        "median_observed_latency_ms_failure_inclusive": average(rows, "latency_ms", median=True),
        "inputs_with_variable_correctness": [i for i, group in inputs.items() if len({r["correct"] for r in group}) > 1],
        "inputs_with_variable_parsed_answer": [i for i, group in inputs.items() if len({json.dumps([r.get("parsed_answer"), r.get("parsed_scale")], sort_keys=True) for r in group if r.get("parsed_answer") is not None}) > 1],
        "successes_per_input": {i: sum(r["correct"] for r in group) for i, group in inputs.items()},
    }


def compare(pairs):
    known = [pair for pair in pairs if all(r.get("total_tokens") is not None for r in pair.values())]
    metrics = {}
    for strategy in ("always_complex", "adaptive"):
        rows = [pair[strategy] for pair in known]
        metrics[strategy] = {
            "mean_total_tokens": average(rows, "total_tokens"),
            "median_latency_ms": average(rows, "latency_ms", median=True),
            "mean_execution_tokens": average(rows, "execution_total_tokens"),
            "median_execution_latency_ms": average(rows, "execution_latency_ms", median=True),
            "mean_router_tokens": average(rows, "router_total_tokens"),
            "median_router_latency_ms": average(rows, "router_latency_ms", median=True),
        }
    reductions = {}
    for field in ("mean_total_tokens", "median_latency_ms", "mean_execution_tokens", "median_execution_latency_ms"):
        baseline, adaptive = metrics["always_complex"][field], metrics["adaptive"][field]
        reductions[field] = (baseline - adaptive) / baseline * 100 if baseline else None
    return {"planned_pairs": len(pairs), "known_usage_pairs": len(known), "excluded_unknown_usage_pairs": len(pairs)-len(known), "strategies": metrics, "adaptive_reduction_percent": reductions}


def main():
    path = Path(sys.argv[1])
    meta = json.loads(path.with_suffix(".run.json").read_text())
    protocol = meta["protocol"]
    for name, expected in {**protocol["sha256"], **meta["sha256"]}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen/run file changed: {name}")
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in events if row["event"] == "result"]
    key = lambda r: (r["id"], r["repetition"], r["strategy"])
    if len(rows) != len({key(r) for r in rows}) or {key(r) for r in rows} != {key(r) for r in meta["schedule"]}:
        raise ValueError("Run incomplete or has duplicate/unexpected results")
    groups = {s: [r for r in rows if r["strategy"] == s] for s in protocol["strategies"]}
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["id"], row["repetition"])][row["strategy"]] = row
    summary = {
        "source": str(path), "scope": protocol["scope"], "complete": True,
        "strategies": {s: describe(group) for s, group in groups.items()},
        "paired_known_usage_comparison": compare(list(pairs.values())),
        "unknown_usage_pairs": [list(k) for k, pair in pairs.items() if any(r.get("total_tokens") is None for r in pair.values())],
        "categories": {},
    }
    for category in sorted({r["id"].split("-", 1)[1] for r in rows}):
        category_pairs = [pair for (i, rep), pair in pairs.items() if i.endswith("-" + category)]
        summary["categories"][category] = {
            "correctness": {s: describe([r for r in group if r["id"].endswith("-"+category)]) for s, group in groups.items()},
            "paired_comparison": compare(category_pairs),
        }
    adaptive = groups["adaptive"]
    decisions = [r for r in adaptive if all(type(r.get(k)) is bool for k in ("needs_retrieval", "needs_reasoning"))]
    matches = [r for r in decisions if all(r[k] == v for k, v in r["expected_route"].items())]
    summary["routing"] = {
        "available_decisions": len(decisions), "unavailable_decisions": len(adaptive)-len(decisions),
        "matching_decisions": len(matches), "match_rate_among_available": len(matches)/len(decisions),
        "mismatches": [{k: r[k] for k in ("id", "repetition", "needs_retrieval", "needs_reasoning", "expected_route", "correct")} for r in decisions if r not in matches],
        "per_capability": {k: {
            "false_positive": sum(r[k] and not r["expected_route"][k] for r in decisions),
            "false_negative": sum(not r[k] and r["expected_route"][k] for r in decisions),
        } for k in ("needs_retrieval", "needs_reasoning")},
    }
    summary["execution_errors"] = [{k: r.get(k) for k in ("id", "repetition", "strategy", "execution_error", "total_tokens")} for r in rows if "execution_error" in r]
    summary["failure_counts"] = {
        "provider_errors_or_timeouts": sum(r.get("execution_error", "").startswith(("InternalServerError:", "APITimeoutError:")) for r in rows),
        "malformed_router_exceptions": sum(r.get("execution_error", "").startswith("KeyError:") for r in rows),
        "invalid_answer_outputs": sum(r["error_type"] in ("format_error", "type_error") for r in rows),
    }
    summary["answer_failures"] = [{k: r.get(k) for k in ("id", "repetition", "strategy", "error_type", "answer_text", "gold_answer", "gold_scale", "needs_retrieval", "needs_reasoning")} for r in rows if not r["correct"]]
    b = summary["strategies"]["always_complex"]["majority_correct_inputs"]
    a = summary["strategies"]["adaptive"]["majority_correct_inputs"]
    comparison = summary["paired_known_usage_comparison"]
    summary["decision"] = {
        "additional_incorrect_adaptive_inputs": b-a,
        "correctness_limit_met": b-a <= protocol["maximum_additional_incorrect_inputs"],
        "overall_cost_success": "inconclusive_unknown_usage" if comparison["excluded_unknown_usage_pairs"] else "met" if all(comparison["adaptive_reduction_percent"][f] >= 10 for f in ("mean_total_tokens", "median_latency_ms")) else "not_met",
        "overall_success_criterion": "not_met_correctness" if b-a > protocol["maximum_additional_incorrect_inputs"] else "inconclusive_unknown_usage" if comparison["excluded_unknown_usage_pairs"] else "see_overall_cost_success",
    }
    output = path.with_suffix(".summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"correct_inputs": {"always_complex": b, "adaptive": a}, "paired_comparison": comparison, "routing": {k: v for k, v in summary["routing"].items() if k != "mismatches"}, "decision": summary["decision"]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
