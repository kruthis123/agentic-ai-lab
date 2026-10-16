# Day 2 — Methods, results and my interpretation

Date: 16 October 2026

Source: [MCP Tool Descriptions Are Smelly! — v3, 31 May 2026](https://arxiv.org/html/2602.14878v3)

Read: Sections 4.2–4.5, 5.1–5.3 and 8.

## What the study does

- The authors select MCP servers from relevant prior studies, not a random sample of the whole ecosystem. They collect tool names, descriptions and input schemas through an MCP client.
- Three LLMs score descriptions. ICC checks agreement between model evaluators; weighted Cohen's kappa checks alignment with human assessments. Agreement does not establish that scores predict successful tool use: evaluators may share blind spots.
- The initial rewrite uses the original description, input schema and rubric. Examples are withheld initially; actual execution traces supply grounded examples and refine limitations. Length and completeness assess the whole description rather than occupying a separate output field.
- Performance is evaluated on MCP-Universe, a subset of the collected tools. For three models, original-description baselines are previously published aggregate results; for Qwen3-Next, both conditions are run by the authors. Different execution conditions could explain part of a difference attributed to descriptions.

## How performance is measured

- **Success rate (SR):** fraction of tasks passing all their evaluators.
- **Average evaluator score (AE):** average fraction of evaluators passed per task; gives partial credit.
- **Average steps (AS):** average execution-step count; a cost indicator, not a direct measurement of tokens, money or time.

Not every setting includes every component: Qwen3-Next omits Examples and Parameter Explanation, and Browser Automation omits Examples because of context-length constraints. Component comparisons remove Examples or test Purpose paired with another component; they are not exhaustive tests of every combination or deliberately poor descriptions.

## What I took from the results

Unstated limitations, missing usage guidance and opaque parameters are common smells under the rubric. Augmentation does not improve success in every model–domain setting, and execution steps generally increase. There is no universally best component combination.

My main takeaway is that a rubric weakness and a performance weakness are not the same thing. A low component score identifies a potential documentation weakness, but does not establish that adding that component will improve agent performance. Its practical value depends on the model, domain and information it contributes. The component comparisons also do not establish that fixing any particular low-scoring component will worsen performance.

## The limitation that matters most to me

Construct validity: does the rubric's definition of “good” capture what actually matters for successful tool use?

The common threshold is a researcher-chosen, subjective assessment rule. Varying performance results give me reason to question its predictive usefulness, but do not directly test whether another threshold would be better. I would treat smell scores as prompts for investigation, not automatic instructions to add detail.

## My current position

Use structure to identify possible gaps, but judge changes by observed behaviour—not rubric scores alone. More detail may help, introduce unnecessary work or make results worse.

Experiment: still undecided. Continue studying before choosing what to test.
