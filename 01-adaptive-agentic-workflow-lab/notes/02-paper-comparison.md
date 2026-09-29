# Comparing Adaptive Agentic Workflows

## References

- Xiang et al., **LLM-as-Scheduler: Agentic Workflow Dynamic Scheduling**, ACL 2026  
  https://aclanthology.org/2026.acl-long.581/
- Su et al., **Difficulty-Aware Agentic Orchestration for Query-Specific Multi-Agent Workflows**, WWW 2026  
  https://arxiv.org/abs/2509.11079
- Wang et al., **X-Router: Decoupling Knowledge and Reasoning for Cost-Effective LLM Inference**, Findings of ACL 2026  
  https://aclanthology.org/2026.findings-acl.994/
- Read on: 29/09/2026

## Comparison

| Dimension | LLM-as-Scheduler (LAS) | DAAO | X-Router |
|---|---|---|---|
| Problem | A fixed workflow runs expensive later stages even when an earlier output is sufficient. | A single workflow topology can be too complex for easy queries or insufficient for difficult ones. | Always enabling both retrieval and reasoning wastes computation and can sometimes reduce answer quality. |
| When adaptation happens | During execution, after observing intermediate artifacts. | Before executing the generated workflow. Training outcomes improve decisions for future queries. | After a lightweight retrieval probe and direct-answer draft, but before executing the selected final pipeline. |
| What is adapted | The remaining route through an existing workflow. | Workflow depth, operators/agents and model assignment. | Whether to use retrieval and whether to use CoT. |
| Main signal | Cheap gate signals followed by an LLM scheduler for promising outputs. | A learned query-difficulty representation. | Retriever-score dispersion (NQC) and draft uncertainty (NLL). |
| Available choices | Early exit, testing, verification, refinement or continuation through the original workflow. | A query-specific DAG assembled from an operator and model pool. | Direct, RAG, CoT or RAG + CoT. |
| Learning | The gate and scheduler learn to route intermediate outputs. | A VAE-based estimator and orchestration policy learn from whether generated workflows solve queries and from utility-cost feedback. | Two lightweight MLP heads learn `need_RAG` and `need_CoT` from cost-aware oracle labels. |
| Cost-quality principle | Skip stages when the expected benefit does not justify continuing. | Maximise expected utility while penalising workflow cost. | Select the pipeline maximising answer quality minus token and latency penalties. |
| Cost of under-processing | Incorrect early exit or skipping a stage required for correctness. | An underestimated query receives a workflow that may be too weak to solve it. | Missing retrieval can remove necessary evidence; missing CoT can omit required multi-step reasoning. |
| Cost of over-processing | Unnecessary calls, tokens and latency. | An overestimated query receives an unnecessarily complex workflow. | Unnecessary retrieval adds cost and distracting context; unnecessary CoT adds tokens and latency and can hurt quality. |

## Important signal definitions

- **NQC:** Normalised dispersion of the top retrieval scores. High NQC means some retrieved documents stand out clearly, indicating stronger retriever commitment. It measures retrieval confidence, not retrieval necessity directly.
- **NLL:** Length-normalised negative log-likelihood of a single direct-answer draft. Higher NLL means lower generation confidence and is used as a proxy for whether more reasoning may help.
- **DAAO difficulty:** A learned estimate used to influence workflow construction. Success or failure updates future estimates; it does not rebuild the current query's workflow during execution.

## Where the papers agree

All three papers argue that applying the most expensive workflow to every query is inefficient. They treat adaptation as a cost-quality decision rather than assuming that more computation is always better. They also show that a wrong routing decision has asymmetric consequences: under-processing can damage correctness, while over-processing often wastes resources and can occasionally reduce quality.

## Important differences

- LAS starts with a strong existing workflow and changes the remaining route using intermediate outputs.
- DAAO constructs a different workflow for each query using a learned difficulty representation before workflow execution begins.
- X-Router does not primarily model difficulty on one axis. It separately estimates whether external evidence and additional reasoning are worth their costs.
- X-Router has a small fixed action space, while DAAO can vary workflow topology, operators and models.

## My current understanding

I initially viewed adaptation mainly as giving easy queries less computation and difficult queries more computation. X-Router changed this view: two queries that appear equally difficult may need different capabilities. One may need external evidence but little reasoning, while another may need substantial reasoning but no retrieval.

LAS, DAAO and X-Router therefore represent three different forms of adaptation: changing the remaining path after observing intermediate work, constructing the whole workflow from a predicted difficulty, and selecting capabilities along separate knowledge and reasoning dimensions. None of these approaches guarantees efficiency. Their value depends on whether the routing signal is reliable, whether the avoided work is more expensive than the routing overhead, and how costly each type of routing error is.

The evidence is also bounded. LAS was evaluated on code and mathematics, DAAO on benchmark task families, and X-Router on six QA benchmarks. X-Router's cross-domain evaluation is useful, but it does not establish generalisation to coding agents, open-ended tool use or long-running workflows.

## Relevance to my experiment

- Compare a fixed expensive strategy with an adaptive strategy under the same model, inputs and evaluation criteria.
- Measure answer quality, tokens, latency and routing overhead instead of reporting cost reduction alone.
- Separate retrieval errors from reasoning errors rather than treating every failure as incorrect difficulty prediction.
- Include both under-processing and over-processing cases.
- Treat improved efficiency as a hypothesis, not a guaranteed result.
- Clearly state that a small experiment is not a reproduction of LAS, DAAO or X-Router.

## Question to carry forward

Is adaptive execution valuable mainly because it changes how much computation a query receives, or because it selects the particular capability that the query is missing?
