# LLM-as-Scheduler: Agentic Workflow Dynamic Scheduling

## Reference

- Xiang et al., ACL 2026
- Paper: https://aclanthology.org/2026.acl-long.581/
- Read on: 28/09/2026

## Problem

Fixed agentic workflows apply expensive verification, refinement and testing stages to every query, even when the first output is already correct.

## Proposed approach

LAS adds adaptive scheduling to an existing workflow:

1. A lightweight gate examines each intermediate output.
2. Clearly poor outputs continue through the original workflow.
3. Promising outputs that may be able to skip later steps are sent to the LLM scheduler.
4. The scheduler selects early exit, testing, verification or refinement.

The gate uses:

- Programmatic specification checks
- A small trained quality classifier
- Agreement between candidates
- Historical agent reliability

## Important definitions

- **Easy:** First-agent output passes the correctness oracle.
- **Hard:** First output fails, but the completed workflow passes.
- **Unsolved:** Both first and final outputs fail.

These labels are retrospective and relative to the evaluated workflow. They are not intrinsic measures of task difficulty.

## Main evidence

Compared with the fixed AFlow workflow:

| Benchmark | Accuracy change | Token change | Latency change |
|---|---:|---:|---:|
| MBPP | −0.5 points | −63.4% | −41.9% |
| HumanEval | −1.4 points | −37.4% | −36.8% |
| GSM8K | −0.8 points | −50.9% | −36.9% |

Important ablation finding:

- Gate only: efficient but significantly less accurate.
- LAS only: preserves accuracy but consumes more tokens.
- Gate + LAS: produces the best quality–efficiency trade-off.

## Limitations

- Evaluated only on code-generation and mathematical-reasoning benchmarks.
- The gate classifier and thresholds are benchmark-specific.
- Generalisation to RAG, tool-using agents and other workflows was not demonstrated.
- LAS assumes a strong existing workflow.
- “Unsolved” means unsolved by this workflow, not impossible.
- Only output-token usage is reported as the token-cost metric.

## My understanding

The inexpensive gate sends clearly poor outputs through the original workflow, while promising outputs go to LAS so it can decide whether to exit early or route them to another useful stage. The most convincing result for me was that Gate + LAS retained nearly all of AFlow's accuracy while substantially reducing output tokens and latency; the ablation also showed why neither component was as effective alone. However, the paper does not establish that the same gains will generalise to all agentic applications because it evaluates only code-generation and mathematical-reasoning benchmarks and uses benchmark-specific gate training and thresholds. I think adaptive scheduling is most useful when a strong but expensive workflow already exists, many queries can be solved before reaching its final stage, and inexpensive signals can identify those queries reliably. The scheduler's overhead may not be justified when the original workflow is already short, most inputs require all stages, or the gate cannot reliably identify promising intermediate outputs. Therefore, I will treat improved efficiency as a hypothesis to test in my experiment rather than an outcome guaranteed by LAS.

## Relevance to my experiment

### Ideas I may test

- Static execution versus adaptive escalation.
- Whether easy cases benefit from additional processing.
- Quality versus calls, tokens and latency.
- The consequences of incorrect early exit.

### Important differences

- I will not train a specialised gate model.
- I will use a smaller, original task set.
- I will not reproduce AFlow or the complete LAS architecture.
- My results will not establish generalisation beyond my experiment.

## Question to carry forward

Can adaptive routing remain useful when correctness cannot be checked easily and the gate has not been trained specifically for the task?
