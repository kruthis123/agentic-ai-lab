"""Sequential frozen evaluation. Dry-run is offline; resume never retries failures."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo

from src import model, routes, strategies
from src.retrieval import TfidfRetriever, load_tatqa_chunks
from src.strategies import run_adaptive, run_always_complex
from src.tatqa_scoring import score_tatqa

PROTOCOL = Path("data/tatqa-evaluation-protocol.json")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["sha256"].items():
        if fingerprint(path) != expected:
            raise ValueError(f"Frozen file changed: {path}")
    if os.getenv("MODEL_NAME") != protocol["model"]:
        raise ValueError("MODEL_NAME does not match the frozen protocol")
    if str(model.client.base_url) != protocol["provider_base_url"]:
        raise ValueError("Provider does not match the frozen protocol")
    if model.client.timeout != protocol["request_timeout_seconds"]:
        raise ValueError("Timeout does not match the frozen protocol")
    if model.client.max_retries != protocol["max_retries"]:
        raise ValueError("Retry setting does not match the frozen protocol")
    queries = [json.loads(line) for line in Path(protocol["inputs_file"]).read_text().splitlines()]
    contexts = json.loads(Path(protocol["contexts_file"]).read_text())
    if len(queries) != protocol["input_count"] or len(contexts) != protocol["context_count"]:
        raise ValueError("Dataset size does not match the protocol")
    rng = random.Random(protocol["seed"])
    schedule = []
    for r in range(protocol["repetitions"]):
        shuffled = queries.copy()
        rng.shuffle(shuffled)
        for i, query in enumerate(shuffled):
            strategies = protocol["strategies"] if (i + r) % 2 == 0 else protocol["strategies"][::-1]
            for strategy in strategies:
                schedule.append({"id": query["id"], "repetition": r + 1, "strategy": strategy})
    return protocol, queries, contexts, schedule


def attempt_key(record):
    return record["id"], record["repetition"], record["strategy"]


def append_record(output, record):
    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    output.flush()
    os.fsync(output.fileno())


def failed_metrics(calls, adaptive, elapsed, execution_start):
    """A failed request has unknown usage; a stage never attempted costs zero."""
    metrics = {"latency_ms": elapsed}
    for prefix, selected in (
        ("", calls),
        ("router_", calls[:1] if adaptive else []),
        ("execution_", calls[1:] if adaptive else calls),
    ):
        for field in ("input_tokens", "output_tokens"):
            values = [call.get(field) for call in selected]
            metrics[prefix + field] = sum(values) if all(v is not None for v in values) else None
        inp, out = metrics[prefix + "input_tokens"], metrics[prefix + "output_tokens"]
        metrics[prefix + "total_tokens"] = inp + out if inp is not None and out is not None else None
    # Include route parsing and retrieval overhead, not just request duration.
    router_time = (execution_start if execution_start is not None else elapsed) if adaptive else 0.0
    metrics["router_latency_ms"] = router_time
    metrics["execution_latency_ms"] = elapsed - router_time
    return metrics


def run_attempt(query, strategy, retriever):
    calls = []
    original = routes.call_model
    original_choose = strategies.choose_route
    execution_start = None
    start = time.perf_counter()

    def tracked_choose(*args, **kwargs):
        nonlocal execution_start
        decision = original_choose(*args, **kwargs)
        execution_start = (time.perf_counter() - start) * 1000
        return decision

    def tracked_call(prompt):
        # Observe existing calls without changing prompts or the frozen routes.
        call = {"started_ms": (time.perf_counter() - start) * 1000}
        calls.append(call)
        try:
            result = original(prompt)
            call.update(asdict(result))
            return result
        except (Exception, KeyboardInterrupt) as error:
            call.update(error=f"{type(error).__name__}: {error}", latency_ms=(time.perf_counter() - start) * 1000 - call["started_ms"])
            raise

    routes.call_model = tracked_call
    strategies.choose_route = tracked_choose
    interrupted = False
    try:
        function = run_adaptive if strategy == "adaptive" else run_always_complex
        result = function(question=query["question"], answer_type="tatqa", provided_context=query["provided_context"], retriever=retriever)
        result.update(score_tatqa(result["answer_text"], query["gold_annotation"]))
        result["status"] = "failed" if result["error_type"] in ("format_error", "type_error") else "completed"
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        result = score_tatqa("", query["gold_annotation"])
        result.update(
            status="interrupted" if interrupted else "failed",
            error_type="interrupted_unknown" if interrupted else "execution_error",
            execution_error=f"{type(error).__name__}: {error}",
            answer_text=None, needs_retrieval=True if strategy == "always_complex" else None,
            needs_reasoning=True if strategy == "always_complex" else None,
        )
        result.update(failed_metrics(calls, strategy == "adaptive", (time.perf_counter() - start) * 1000, execution_start))
        if strategy == "adaptive" and calls and "text" in calls[0]:
            try:
                decision = json.loads(calls[0]["text"])
                for name in ("needs_retrieval", "needs_reasoning"):
                    if type(decision.get(name)) is bool:
                        result[name] = decision[name]
            except (ValueError, AttributeError):
                pass
    finally:
        routes.call_model = original
        strategies.choose_route = original_choose
    result["model_calls"] = calls
    return result, interrupted


def recover(output_path, output):
    """A start without a saved result is unknown, not a reason to repeat the call."""
    events = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    done = {attempt_key(row) for row in events if row["event"] == "result"}
    for row in events:
        key = attempt_key(row)
        if row["event"] == "started" and key not in done:
            append_record(output, {**row, "event": "result", "status": "interrupted", "error_type": "interrupted_unknown", "correct": False, "exact_match": 0.0, "f1": 0.0, "answer_text": None, "input_tokens": None, "output_tokens": None, "total_tokens": None, "latency_ms": None})
            done.add(key)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup and schedule; no calls or result files")
    parser.add_argument("--resume", type=Path, help="Continue the same JSONL run, skipping saved failures too")
    parser.add_argument("--limit", type=int, help="Stop after this many new attempts; resume later")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    protocol, queries, contexts, schedule = prepare()
    print(f"Verified frozen files/settings: {len(queries)} inputs, {len(schedule)} attempts; no retries.")
    if args.dry_run:
        print("Dry-run only: no API calls and no result files created.")
        return
    code_hashes = {path: fingerprint(path) for path in (str(PROTOCOL), "src/model.py", "src/strategies.py", "src/run_tatqa_evaluation.py")}
    if args.resume:
        output_path = args.resume
        metadata = json.loads(output_path.with_suffix(".run.json").read_text())
        if metadata["sha256"] != code_hashes or metadata["schedule"] != schedule:
            raise ValueError("Run code/protocol/schedule changed; cannot resume")
        if not output_path.exists():
            raise ValueError("Missing run output; cannot resume")
    else:
        stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S-%f")
        output_path = Path(f"results/tatqa-evaluation-{stamp}.jsonl")
        output_path.parent.mkdir(exist_ok=True)
        metadata = {"run_id": output_path.stem, "protocol": protocol, "sha256": code_hashes, "schedule": schedule}
        output_path.with_suffix(".run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        output_path.touch(exist_ok=False)
    retrievers = {c["context_id"]: TfidfRetriever(load_tatqa_chunks(c)) for c in contexts}
    by_id = {q["id"]: q for q in queries}
    print(f"Saving to {output_path}", flush=True)
    attempted = 0
    with output_path.open("a", encoding="utf-8") as output:
        done = recover(output_path, output)
        for item in schedule:
            if attempt_key(item) in done:
                continue
            query = by_id[item["id"]]
            record = {**item, "run_id": metadata["run_id"], "context_id": query["context_id"], "condition": query["condition"], "question": query["question"], "expected_route": query["expected_route"], "gold_answer": query["gold_annotation"]["answer"], "gold_scale": query["gold_annotation"]["scale"], "model": protocol["model"]}
            append_record(output, {**record, "event": "started"})
            result, interrupted = run_attempt(query, item["strategy"], retrievers[query["context_id"]])
            append_record(output, {**record, **result, "event": "result"})
            attempted += 1
            print(f"{item['id']} / repetition {item['repetition']} / {item['strategy']}: {result['status']}, correct={result['correct']}", flush=True)
            if interrupted or (args.limit is not None and attempted >= args.limit):
                break
    print(f"Saved {attempted} new attempts. Resume with --resume {output_path}")


if __name__ == "__main__":
    main()
