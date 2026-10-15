# Day 1 — Tool description smells

Date: 14 October 2026

Source: [MCP Tool Descriptions Are Smelly! — v3, 31 May 2026](https://arxiv.org/html/2602.14878v3)

Read: Abstract, Sections 1–2, 3.1, opening of 3.2, and 4.1.

## My understanding

Tool descriptions influence how an agent selects tools and supplies arguments. Ambiguities may remain unnoticed until the agent encounters requests not covered during development testing. A structured approach may improve guidance, but its effect on task success and efficiency needs to be measured; a high rubric score does not itself prove better performance.

The model receives the tool name, description and input schema. It proposes a call; the MCP client sends the request, and the server runs the tool implementation. Descriptions guide behaviour, while software must enforce input rules, permissions and safety restrictions.

## Example that made it clear

In the stock-price example, unclear date-parameter names and formats may lead the model to request a broader period than needed. Explicit date arguments help it construct a bounded request. This is primarily an argument-construction problem, not choosing the wrong tool. The example illustrates a mechanism; establishing causation requires a controlled comparison.

## The six components

| Component | My explanation |
| --- | --- |
| Purpose | What the tool does. |
| Guidelines | When and how to use it. |
| Limitations | Constraints, caveats and situations where it may fail or be less effective. |
| Parameter explanations | What inputs mean and how to use them, considered alongside the input schema. |
| Length and completeness | Enough detail for the tool's complexity—not simply a longer description. |
| Examples | Illustrations of correct and effective use. |

The authors combined official guidance with an LLM-assisted search of community sources. Two authors independently identified recurring recommendations and measured their agreement. The framework came from source analysis, not performance experiments.

Each component is scored from 1 to 5: below 3 indicates a smell; 3 is the minimum acceptable score. A smell signals a potential quality weakness, not a proven execution fault. Later, the scanner applies this threshold to the average of three model evaluators' scores.

## My provisional view

Every tool may not need equally detailed coverage of all components. Information can already be clear elsewhere without requiring a separate heading. However, a simple tool can still have important limitations. This is my interpretation to investigate.

## Questions for Day 2

1. Does meeting the minimum score across all six components consistently improve agent performance?
2. How does weakness in each component affect behaviour, and do some weaknesses matter more? Removing a component is not necessarily equivalent to making it misleading or inadequate.
3. What trade-offs arise from adding detail, including more tokens, extra steps or worse results?

Experiment: undecided. Study the evidence before choosing what to test.
