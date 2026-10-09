# Retrieval Qualified Routing Evaluation Protocol

Frozen on 9 October 2026, after development and before model evaluation on this set. This amendment supplements [the original design](03-experiment-design.md); it does not replace it or make the Day 5 findings part of the final evaluation.

## Question and scope

When the retriever can return the required evidence, how does adaptive capability selection compare with always using retrieval and a reasoning instruction, in correctness, tokens and latency?

The set has 76 inputs from 38 source questions in 19 TAT-QA development-split contexts: 19 inputs per expected route. It is the retrieval-qualified subset of the original seed-42 selection, after five documented question refinements. Six contexts were excluded because at least one question missed its required evidence; no replacement contexts were sampled. Results apply to this controlled subset, not retrieval reliability, the full TAT-QA benchmark, or agentic systems generally.

## Fixed setup

- Inputs: `data/tatqa-evaluation-qualified.jsonl`; documents: `data/tatqa-evaluation-qualified-contexts.json`.
- Compare only `adaptive` and `always_complex`; three repetitions per input and strategy. This is 456 strategy executions, normally 684 model calls without failures.
- Model: `gemma4:26b` at the existing NUS endpoint; temperature 0; JSON-object response mode; the same model for routing and answering. Temperature 0 does not guarantee identical responses.
- Keep the existing prompts, reasoning instruction, TF-IDF chunking/ranking, top-k of 2, and official TAT-QA scoring unchanged. The reasoning condition is an added instruction, not a claim about hidden reasoning.
- Shuffle input order each repetition using one RNG seeded with 42. At each input, alternate which strategy runs first by shuffled position and repetition. Execute sequentially, never concurrently.
- Use a 90-second SDK timeout per model request and zero automatic retries, identically for both strategies. Task 4 must implement these transport settings before evaluation starts; prompt/scoring hashes remain fixed.
- Index construction, scoring and file writes are outside strategy latency. Retrieval execution and answer generation are inside execution latency; the adaptive router is also inside end-to-end latency.
- Send only the question and supplied context to the router, and question/context/retrieved evidence to the answer model. Gold answers, source annotations, expected labels and qualification records stay outside prompts.

## Correctness and metrics

Use the existing official scale-normalized TAT-QA exact match and F1. An input is correct if at least two of its three executions pass exact match. Report this input-level correctness out of 76 and raw execution accuracy out of 228 per strategy, overall and by category. Do not manually override scores.

Primary comparison: mean end-to-end total tokens and median end-to-end latency, including router overhead. Diagnostic comparison: mean execution-only tokens and median execution-only latency. Also report router costs, expected-route matches, false positives/negatives, scale accuracy, response variability and failure counts. Literal scale accuracy is not an independent unit-validity verdict: percentage strings with source scale `""` can match a numeric answer with scale `percent` while that scale flag is false.

Calculate a reduction as `(baseline - adaptive) / baseline * 100`; a negative reduction means an increase. Compare costs on the same paired input/repetition IDs. Cost eligibility depends on known usage, not correctness: keep wrong answers and invalid outputs in the cost comparison when their usage is known. With no unknown-cost failures, all 228 pairs are included.

## Decision rule

Retain the original efficiency targets: at least 10% reductions in both mean end-to-end tokens and median end-to-end latency. Adaptive may have no more than one additional incorrect input compared with always-complex. Execution-only savings or route-label accuracy alone do not satisfy the end-to-end success criterion.

If either efficiency target or the correctness limit fails, the criterion is not met for this setup. If unknown-cost provider failures prevent a complete cost comparison, overall efficiency success is inconclusive rather than assumed. This is a small descriptive comparison, not proof of general superiority or statistical equivalence.

## Failures and resume rules

Record invalid JSON, invalid types, malformed router decisions, timeouts and provider errors as failed executions. Do not repair responses, silently choose a fallback route, rerun failed attempts, or select the best response. Preserve partial responses/known usage and actual observed failure latency where available; unknown usage is null, never zero.

For provider-failure pairs, report completed-pair token/latency diagnostics separately from failure-inclusive correctness and observed end-to-end latency. State the number of excluded pairs and that total token cost is unknown; do not present conditional cost averages as full-run costs.

Save each attempt immediately, identified by run ID, input ID, repetition and strategy. Resume only IDs absent from the saved run, not IDs already recorded as failures. An attempt interrupted before it can be saved is marked interrupted/unknown, not silently regenerated. Do not change data, prompts, labels, model, retrieval, scoring or thresholds after evaluation starts; necessary changes require an explicitly new run/protocol version, preserving the old records.

## Frozen files

Dataset and prompt/retriever/scorer SHA-256 fingerprints are in `data/tatqa-evaluation-protocol.json`. Verify them and the model/transport settings before starting or resuming. Task 4 adds the evaluation runner and failure recording without changing these frozen components. The original selection, audit and exclusions remain available for provenance.
