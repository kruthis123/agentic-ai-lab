# Day 6 — Evaluation results

All 456 planned strategy attempts completed: 76 inputs, two strategies, three repetitions. Dataset, prompts, scorer and run-code fingerprints were verified before analysis. No failed attempts were retried and no answers were manually rescored.

Run: `results/tatqa-evaluation-20261009-122017-244501.jsonl`. Machine-readable analysis: the same stem with `.summary.json`.

## Main result

Adaptive routing did **not** meet my frozen success criterion in this setup.

| Metric | Always-complex | Adaptive |
| --- | ---: | ---: |
| Correct inputs (at least 2/3 repetitions) | 53/76 (69.7%) | 51/76 (67.1%) |
| Correct executions, failures included | 160/228 | 155/228 |
| Mean total tokens, 225 known-usage pairs | 1,093.6 | 1,324.4 |
| Median total latency, same 225 pairs | 9.62 s | 21.14 s |
| Mean answer-stage tokens, same pairs | 1,093.6 | 811.7 |
| Median answer-stage latency, same pairs | 9.62 s | 10.64 s |

Adaptive had two additional incorrect inputs; I allowed only one. On known-usage pairs, total tokens increased 21.1% and median total latency increased 119.7%. Excluding the router, answer-stage tokens decreased 25.8%, but median answer-stage latency increased 10.5%.

Three of the 228 pairs have unknown total usage because of provider failures/timeouts. Full-run token efficiency is therefore inconclusive, rather than estimated by filling missing usage with zero. The known-usage comparison includes incorrect answers, invalid outputs and malformed-router failures when usage is known; it is not a correct-answers-only comparison. Across all 228 executions per strategy, including observed provider-failure latency, median total latency was 10.24 s versus 21.29 s.

## Where savings and mistakes occurred

Each category has 19 inputs. Majority-correct counts:

| Category | Always-complex | Adaptive | Answer-stage token reduction* |
| --- | ---: | ---: | ---: |
| Lookup, supplied context | 19 | 19 | 38.0% |
| Lookup, retrieval needed | 18 | 18 | 3.7% |
| Arithmetic, supplied context | 8 | 6 | 39.8% |
| Arithmetic, retrieval needed | 8 | 8 | 2.1% |

*Computed on known-usage pairs within each category (57, 57, 55 and 56 respectively). Arithmetic-supplied savings include three malformed-router attempts that never produced an answer. This diagnostic is not evidence that a free, reliable classifier would achieve the same result.

The largest answer-stage token savings were on supplied-context inputs, not retrieval-plus-reasoning inputs as I initially expected. Always-complex retrieves again even when the prompt already contains evidence; avoiding this duplicated context plausibly explains part of the difference, but this experiment does not isolate that mechanism causally.

## Routing and failure audit

- 223/228 adaptive attempts produced usable route decisions; 213/223 matched the expected labels (95.5%). Across all attempts, that is 213/228 (93.4%). These are capability labels, not proven optimal routes.
- Retrieval: three unnecessary retrieval decisions, no observed missed-retrieval decisions among available decisions.
- Reasoning: four unnecessary reasoning decisions, three missed-reasoning decisions. All three misses were `E11-arithmetic-retrieval`, and all three answers were incorrect. This is an association, not proof the routing decision caused the error.
- Seven caught execution exceptions were previously described loosely as request errors. Precisely: **three provider failures/timeouts and four malformed-router exceptions**. The latter returned misspelled keys and string booleans. There were also 21 invalid answer outputs (9 baseline, 12 adaptive).
- `E1-arithmetic-supplied` failed router parsing in all three adaptive repetitions while baseline was correct in all three. A router must produce a valid decision, not merely consume few tokens.
- Baseline-only majority successes: `E1`, `E11`, `E14`, `E18` arithmetic-supplied. Adaptive-only majority successes: `E3`, `E12` arithmetic-supplied. Net difference: two inputs, not four.
- Correctness varied across repetitions for three baseline inputs and four adaptive inputs despite temperature 0. Parsed answers/scales varied for eight baseline inputs and seven adaptive inputs (excluding missing parsed answers).

My hypothesis that missing retrieval would be the most damaging routing error was not tested by an observed example here: none occurred among available decisions. Malformed routing and missed reasoning were the visible routing problems; arithmetic/answer correctness also failed when expected capabilities were selected.

## My defensible takeaway

In this small, retrieval-qualified financial-QA experiment, selecting fewer capabilities reduced answer-stage tokens, but using the same LLM as a sequential router added enough overhead to erase those savings. High capability-label accuracy did not guarantee better end-to-end efficiency or preserved correctness.

This does not show that adaptive workflows generally fail, nor reproduce X-Router or its classifier. Removing router overhead is only a diagnostic; a real classifier would have its own latency, implementation and error profile. The experiment compares retrieval plus an added reasoning instruction, not measured hidden reasoning. Nineteen shared contexts, paired evidence variants, question refinements, retrieval qualification and one model/provider limit generalization. Provider tokens are reported usage, not independently measured compute or monetary cost.

## Reproduce the analysis (no API calls)

```sh
uv run python -m src.summarize_tatqa_evaluation results/tatqa-evaluation-20261009-122017-244501.jsonl
```

Keep these negative results for the post. Any cheaper-router or revised-prompt experiment should be a new, separately recorded experiment, not a change to this completed run.
