# Does every AI question need the same steps?

This is a small learning experiment inspired by [X-Router](https://aclanthology.org/2026.findings-acl.994/). I wanted to find out whether an AI system could save work by choosing which steps each question needs, instead of always doing everything.

## What I compared

I built two ways of answering the same questions:

- **Always use both steps:** search the available document for information, then ask the model to work through the relevant facts or calculations before answering.
- **Choose the steps first:** ask the model whether it needs to search for information, use the extra thinking instruction, do both, or answer directly. Then run the chosen steps.

The model making this choice is called the **router**. It uses an extra AI call before the answer is generated. Both approaches use the same model for answering.

Here, “extra thinking” means an added instruction to work through the facts, check calculations and return the answer. It does not mean that I measured what the model thinks internally.

## What I tested

I used financial questions from the public TAT-QA dataset. Some ask for a fact from a document; others require a calculation.

There were 38 source questions from 19 documents. Each question had two versions: one with the relevant information already supplied and one where the system had to find it. That gave me **76 inputs**, not 76 unrelated questions.

Before running the comparison, I checked that the document search could find the required information. Questions that failed this check were excluded. This lets me focus on choosing steps, but means the results do not represent the full dataset.

I ran each input three times with each approach. An input counted as correct if its answer matched the dataset's expected answer in at least two of the three runs. Failed attempts counted as incorrect.

## Results

Tokens are the small pieces of text the model processes and generates. I counted both the text sent to it and the text it returned.

| Measure | Always use both steps | Choose the steps first |
| --- | ---: | ---: |
| Inputs answered correctly | 53 of 76 | 51 of 76 |
| Average tokens for answering only | 1,094 | 812 |
| Average tokens, including the router | 1,094 | 1,324 |
| Middle response time, including the router | 9.62 seconds | 21.14 seconds |

The token and time figures above compare the same **225 pairs of runs**. Three pairs were left out because service failures meant their token usage was unknown. I did not treat missing usage as zero.

Choosing the steps used about **26% fewer tokens for answering**. But including the router used about **21% more tokens overall**, took longer and answered two fewer inputs correctly. Even without the router, the answering step was not faster.

## What I learned

Skipping unnecessary work can save resources, but deciding what to skip also has a cost. In my small workflow, the extra AI call cost more than it saved, so this approach did not meet my goal.

A cheaper way to choose steps might change the result, but I did not test that here. These findings describe my implementation—not proof that adaptive AI workflows are good or bad in general, and not a reproduction of the paper's system.

## Explore the experiment

- [Code](src/): how the two approaches work.
- [Detailed results and limitations](notes/06-evaluation-results.md).
- [Saved results](results/tatqa-evaluation-20261009-122017-244501.summary.json).
- [Experiment rules](notes/04-evaluation-protocol.md): what I fixed before running the comparison.

To recalculate the summary from the saved results, run this from this folder. It does not make new AI calls:

```sh
uv run python -m src.summarize_tatqa_evaluation results/tatqa-evaluation-20261009-122017-244501.jsonl
```
