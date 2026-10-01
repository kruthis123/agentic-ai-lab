# Experiment Design: Capability-Aware Routing

**Designed on:** 01/10/2026  
**Status:** Pre-registered before implementation; no experimental results have been observed.

This design and its hypotheses were fixed before implementation and before observing any experimental results.

## Research question

On a set of queries from a single domain with different external-evidence and reasoning requirements, how does an adaptive four-route strategy compare with an always-RAG-plus-reasoning baseline in answer correctness, total token usage and end-to-end latency?

## Operational definitions

- **Capability-aware routing:** Selecting among direct, retrieval, reasoning and retrieval-plus-reasoning pipelines based on the estimated value and cost of each capability.
- **Knowledge need:** External evidence not contained in the query is required to produce the correct answer.
- **Reasoning need:** Available information must be transformed or combined, rather than directly extracted, to produce the correct answer.
- **Routing overhead:** The additional model calls, tokens and elapsed time required to select a route.
- **Correct answer:** An answer that matches a predefined gold answer according to a scoring rule fixed before execution.

## Strategies being compared

### Adaptive strategy

The router receives only the query and returns two booleans:

```json
{
  "needs_retrieval": true,
  "needs_reasoning": false
}
```

| Retrieval | Reasoning | Selected route |
|---:|---:|---|
| No | No | Direct |
| Yes | No | Retrieval only |
| No | Yes | Reasoning only |
| Yes | Yes | Retrieval + reasoning |

Router tokens and latency are included in the adaptive strategy's total cost.

### Always-complex baseline

Every query uses retrieval and the reasoning instruction. It does not make a router call. The model, model settings, retriever, document collection, answer schema and evaluation rules remain the same as in the adaptive strategy.

### Reasoning condition

The reasoning route adds a fixed instruction asking the model to identify the relevant facts, apply the required rules or calculations, verify the result and then return only the structured final answer. The direct route does not contain this decomposition instruction.

This is a reasoning-prompt condition; it may encourage additional processing but does not prove or expose the model's internal reasoning process.

## Pilot dataset

The pilot uses the fictional Northstar Labs travel-and-expense domain. Its local policy document provides facts about accommodation limits, meals, flight booking and expense receipts. Fictional facts reduce the chance that the model can answer knowledge-dependent questions from parametric memory.

The 12 pilot queries are balanced across the four expected routes:

| ID | Expected route | Scenario | Answer type | Gold answer |
|---|---|---|---|---|
| D1 | Direct | Extract the destination from Maya's request. | Text | `Tokyo` |
| D2 | Direct | Extract the listed hotel amount. | Numeric | `420` |
| D3 | Direct | Extract Tina's travel date. | Date | `2026-09-01` |
| K1 | Retrieval | Retrieve the London hotel-rate limit. | Numeric | `280` |
| K2 | Retrieval | Retrieve the amount above which a receipt is required. | Numeric | `30` |
| K3 | Retrieval | Retrieve the approval required for booking fewer than 14 days before departure. | Text | `director approval` |
| C1 | Reasoning | Add a SGD 620 flight and three SGD 180 hotel nights. | Numeric | `1160` |
| C2 | Reasoning | Apply a SGD 75 daily meal limit to expenses of 60, 82 and 71. | Numeric | `206` |
| C3 | Reasoning | Calculate out-of-pocket hotel cost for five nights at 250 with a supplied limit of 220. | Numeric | `150` |
| B1 | Retrieval + reasoning | Retrieve London's hotel limit and apply it to four nights at 310. | Numeric | `1120` |
| B2 | Retrieval + reasoning | Retrieve the receipt rule and apply it to labelled expenses E1=12, E2=38 and E3=27. | Set | `["E2"]` |
| B3 | Retrieval + reasoning | Retrieve Tokyo hotel and meal limits, then calculate reimbursement for two nights at 300 and two meal days at 60. | Numeric | `560` |

These are designed route labels, not proof that the labelled route is empirically optimal. The pilot will be used to find implementation or dataset defects before constructing the final 24-query evaluation set.

## Deterministic scoring

Every route must return structured JSON. Scoring does not use an LLM judge.

- Text answers are trimmed, lowercased and compared exactly.
- Numeric answers are parsed and compared using a fixed tolerance of `0.01`.
- Dates are converted to ISO `YYYY-MM-DD` format and compared exactly.
- Sets are compared without considering order; missing or additional elements are incorrect.
- Invalid JSON is incorrect and is also recorded as a format error.
- Scores will not be manually overridden after observing responses.

Each query and strategy will be executed three times. A query is correct when at least two of its three executions are correct. Raw execution-level accuracy will also be reported so that instability remains visible.

## Pre-registered hypotheses

### H1 — Efficiency

Including routing overhead, I expect the adaptive strategy to reduce mean total token usage per query and median end-to-end latency by at least 10% compared with the always-complex baseline.

### H2 — Correctness

Across the final 24-query evaluation set, I expect the adaptive strategy to produce no more than one additional incorrect query compared with the always-complex baseline.

### H3 — Category effect

I expect the largest token and latency reductions in the direct category because the adaptive strategy can avoid both retrieval and the reasoning instruction. I expect the smallest reduction, or a possible increase, in the retrieval-plus-reasoning category because both strategies execute the complex route while the adaptive strategy also pays routing overhead.

### H4 — Routing failures

I expect the most damaging routing failure for answer correctness to be a retrieval false negative: skipping retrieval when external evidence is required and producing an incorrect or ungrounded answer.

## Metrics

Primary metrics:

- Query-level correctness
- Mean total token usage per query, including routing overhead
- Median end-to-end latency, including routing overhead

Diagnostic metrics:

- Expected route and selected route
- Retrieval and reasoning false positives and false negatives
- Router input and output tokens
- Final-generation input and output tokens
- Number of model calls
- Raw execution-level correctness
- Format errors

Efficiency reduction will be calculated as:

```text
(baseline metric - adaptive metric) / baseline metric * 100
```

## Controls

- Same model and model parameters
- Same retriever and retrieval `top_k`
- Same policy-document corpus
- Same query set
- Same final-answer schema and scoring rules
- Same fixed reasoning instruction
- Three executions per query and strategy
- No prompt, label, hypothesis or scoring changes after final evaluation begins

## Failure condition

I will conclude that adaptive routing was not worthwhile for this experiment if it produces more than one additional incorrect query, or if it fails to reduce both mean total token usage and median end-to-end latency by at least 10% compared with the always-complex baseline.

Route-selection accuracy will be reported as a diagnostic, but it will not independently determine success because a route different from the designed label may still produce a correct answer at lower cost.

## Known limitations

- The small synthetic dataset will not establish generalisation beyond this experiment.
- The designed route labels reflect assumptions that the pilot may challenge.
- A reasoning prompt does not prove that the model used a particular internal reasoning process.
- Token and latency results will depend on the selected model, provider and runtime conditions.
- Repeated executions of the same query are not equivalent to additional independent queries.
- This experiment is inspired by LAS, DAAO and X-Router but is not a reproduction of any of them.
