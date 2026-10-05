from dataclasses import dataclass

import json

from src.model import call_model, ModelResult
from src.retrieval import retriever

@dataclass
class RouterResult:
    needs_retrieval: bool
    needs_reasoning: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float

def choose_route(question: str) -> RouterResult:
    prompt = f"""
    Decide which of the following capabilities are required to answer the given question.

    needs_retrieval:
    True only when external policy information not included in the question
    is required.

    needs_reasoning:
    True only when facts must be combined, transformed or calculated rather
    than directly extracted.

    Question:
    {question}

    Return only a JSON object with exactly these two keys:
    needs_retrieval and needs_reasoning.
    Both values must be JSON booleans.
    """
    model_result = call_model(prompt=prompt)
    response = json.loads(model_result.text)

    if not isinstance(response["needs_retrieval"], bool):
        raise ValueError("needs_retrieval must be a JSON boolean")
    if not isinstance(response["needs_reasoning"], bool):
        raise ValueError("needs_reasoning must be a JSON boolean")

    return RouterResult(
        needs_retrieval=response["needs_retrieval"],
        needs_reasoning=response["needs_reasoning"],
        input_tokens=model_result.input_tokens,
        output_tokens=model_result.output_tokens,
        latency_ms=model_result.latency_ms
    )


def execute_route(
    question: str,
    answer_type: str,
    needs_retrieval: bool,
    needs_reasoning: bool,
) -> ModelResult:
    answer_formats = {
        "text": "a JSON string",
        "numeric": "a JSON number",
        "date": "a JSON string in YYYY-MM-DD format",
        "set": "a JSON array of strings",
    }

    if needs_retrieval:
        chunks = retriever.retrieve(query=question)
        context = "\n\n".join(chunk.text for chunk in chunks)
    else:
        context = ""

    if needs_reasoning:
        reasoning_instruction = """
        Identify the relevant facts, apply the required rules or calculations,
        verify the result, and then return only the structured final answer.
        """
    else:
        reasoning_instruction = ""

    prompt = f"""
    Answer the following question. If reference context is supplied, use only that context for policy facts.

    Question:
    {question}

    Reference context:
    {context}

    Additional instruction:
    {reasoning_instruction}

    The answer type is: {answer_type}

    Return only a JSON object with exactly one key named "answer".
    Its value must be {answer_formats[answer_type]}.
    """

    return call_model(prompt)
