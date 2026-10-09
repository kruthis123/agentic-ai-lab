from src.routes import execute_route, choose_route
from src.retrieval import TfidfRetriever, retriever as default_retriever

import time

def token_sum(*counts):
    # Missing provider usage is unknown, not zero or an answer failure.
    return None if any(count is None for count in counts) else sum(counts)

def run_adaptive(
    question: str,
    answer_type: str,
    provided_context: str = "",
    retriever: TfidfRetriever = default_retriever,
):
    start = time.perf_counter()
    route = choose_route(question, provided_context=provided_context)
    execution_start = time.perf_counter()
    result = execute_route(
        question=question,
        answer_type=answer_type,
        needs_retrieval=route.needs_retrieval,
        needs_reasoning=route.needs_reasoning,
        provided_context=provided_context,
        retriever=retriever,
    )
    end = time.perf_counter()

    total_input_tokens = token_sum(route.input_tokens, result.input_tokens)
    total_output_tokens = token_sum(route.output_tokens, result.output_tokens)
    total_tokens = token_sum(total_input_tokens, total_output_tokens)
    total_latency_ms = (end - start) * 1000

    return {
        "strategy": "adaptive",
        "answer_text": result.text,
        "needs_retrieval": route.needs_retrieval,
        "needs_reasoning": route.needs_reasoning,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": total_latency_ms,
        "router_input_tokens": route.input_tokens,
        "router_output_tokens": route.output_tokens,
        "router_total_tokens": token_sum(route.input_tokens, route.output_tokens),
        "router_latency_ms": (execution_start - start) * 1000,
        "execution_input_tokens": result.input_tokens,
        "execution_output_tokens": result.output_tokens,
        "execution_total_tokens": token_sum(result.input_tokens, result.output_tokens),
        "execution_latency_ms": (end - execution_start) * 1000,
    }


def run_fixed(
    question: str,
    answer_type: str,
    needs_retrieval: bool,
    needs_reasoning: bool,
    provided_context: str = "",
    retriever: TfidfRetriever = default_retriever,
):
    strategy_names = {
        (False, False): "always_direct",
        (True, False): "always_retrieval",
        (False, True): "always_reasoning",
        (True, True): "always_complex",
    }
    start = time.perf_counter()
    result = execute_route(
        question=question,
        answer_type=answer_type,
        needs_retrieval=needs_retrieval,
        needs_reasoning=needs_reasoning,
        provided_context=provided_context,
        retriever=retriever,
    )
    end = time.perf_counter()

    return {
        "strategy": strategy_names[(needs_retrieval, needs_reasoning)],
        "answer_text": result.text,
        "needs_retrieval": needs_retrieval,
        "needs_reasoning": needs_reasoning,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": token_sum(result.input_tokens, result.output_tokens),
        "latency_ms": (end - start) * 1000,
        "router_input_tokens": 0,
        "router_output_tokens": 0,
        "router_total_tokens": 0,
        "router_latency_ms": 0.0,
        "execution_input_tokens": result.input_tokens,
        "execution_output_tokens": result.output_tokens,
        "execution_total_tokens": token_sum(result.input_tokens, result.output_tokens),
        "execution_latency_ms": (end - start) * 1000,
    }


def run_always_complex(
    question: str,
    answer_type: str,
    provided_context: str = "",
    retriever: TfidfRetriever = default_retriever,
):
    return run_fixed(
        question=question,
        answer_type=answer_type,
        needs_retrieval=True,
        needs_reasoning=True,
        provided_context=provided_context,
        retriever=retriever,
    )
