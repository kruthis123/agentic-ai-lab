# Day 6 — Runner ready (Task 4)

Implemented and checked offline. No evaluation API calls have been made by this task, and there are no measured evaluation findings yet.

From the lab directory:

```sh
uv run python -m src.run_tatqa_evaluation --dry-run
```

Task 5 starts the actual evaluation:

```sh
uv run python -m src.run_tatqa_evaluation
```

The runner prints its result path. To continue that same run:

```sh
uv run python -m src.run_tatqa_evaluation --resume results/tatqa-evaluation-<timestamp>.jsonl
```

Optional `--limit 10` stops after ten new strategy attempts; resume the same file later. It does not change the selection or schedule. Without `--resume`, a new run is created.

## What is saved

- A `.run.json` sidecar holds the complete frozen protocol, schedule and runner/model/strategy fingerprints. Resume rejects changed files or settings.
- The JSONL is an event log: `event: started` is saved before an attempt; `event: result` is saved after it. **Analyze only result events**, not start events. A complete run normally has 456 result events and 456 start events.
- Each result includes its input ID, repetition, strategy, score, metrics and raw model responses. Invalid outputs and provider failures are retained; unknown token usage is null. Successful router usage survives a failed answer request.
- Resume skips all saved results, including failures. A start without a result is recorded as interrupted/unknown and skipped. Ctrl+C during a model attempt saves an interrupted result before stopping; an abrupt process exit may leave only a start event, with no recoverable latency/usage.
- A malformed/truncated log is rejected for inspection rather than silently repaired or retried. Do not run two writers against the same run file.

The client now uses a 90-second SDK timeout and zero automatic retries, checked against the [official Python SDK documentation](https://developers.openai.com/api/reference/python). This is an SDK request timeout, not a guaranteed 90-second wall-clock deadline for the whole adaptive strategy (which can make two requests).

Offline checks covered the deterministic 456-attempt schedule (114 pairs with each strategy first), successful scoring, malformed router output, partial usage after an answer timeout, interruption and failure-skipping resume. Prompts, retrieval, scoring and the qualified dataset remain unchanged. No new unit-test files were added.
