from src.routes import execute_route, choose_route

import time

def run_adaptive(question: str, answer_type: str):
    start = time.perf_counter()
    route = choose_route(question)
    result = execute_route(
        question=question,
        answer_type=answer_type,
        needs_retrieval=route.needs_retrieval,
        needs_reasoning=route.needs_reasoning,
    )
    end = time.perf_counter()

    total_input_tokens = (
        route.input_tokens
        + result.input_tokens
    )
    total_output_tokens = (
        route.output_tokens
        + result.output_tokens
    )
    total_tokens = total_input_tokens + total_output_tokens
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
    }


def run_always_complex(question: str, answer_type: str):
    start = time.perf_counter()
    result = execute_route(
        question=question,
        answer_type=answer_type,
        needs_retrieval=True,
        needs_reasoning=True,
    )
    end = time.perf_counter()

    return {
        "strategy": "always_complex",
        "answer_text": result.text,
        "needs_retrieval": True,
        "needs_reasoning": True,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
        "latency_ms": (end - start) * 1000,
    }
