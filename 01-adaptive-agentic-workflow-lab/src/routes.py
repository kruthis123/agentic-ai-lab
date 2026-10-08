from dataclasses import dataclass

import json

from src.model import call_model, ModelResult
from src.retrieval import TfidfRetriever, retriever as default_retriever

@dataclass
class RouterResult:
    needs_retrieval: bool
    needs_reasoning: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float

def choose_route(question: str, provided_context: str = "") -> RouterResult:
    prompt = f"""
    Decide which of the following capabilities are required to answer the given question.

    needs_retrieval:
    True only when evidence required to answer the question is missing from
    both the question and the supplied context. Retrieval searches the selected
    document. If the supplied context already contains the necessary evidence,
    set needs_retrieval to false.

    needs_reasoning:
    True only when facts must be combined, transformed or calculated rather
    than directly extracted.

    Question:
    {question}

    Supplied context:
    {provided_context}

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
    provided_context: str = "",
    retriever: TfidfRetriever = default_retriever,
) -> ModelResult:
    answer_formats = {
        "text": "a JSON string",
        "numeric": "a JSON number",
        "date": "a JSON string in YYYY-MM-DD format",
        "set": "a JSON array of strings",
    }

    if answer_type == "tatqa":
        output_instruction = """
        Return only a JSON object with exactly two keys: "answer" and "scale".
        The answer must be a number, a string, or an array for multiple answers.
        Infer the scale from the document evidence. Allowed scales are:
        "", "thousand", "million", "billion", "percent".
        Use "" for answers with no scale, including non-numeric text.
        Return numbers in the stated scale without currency symbols or commas.
        For percentages, return the percentage value with scale "percent".
        """
    else:
        output_instruction = f"""
        Return only a JSON object with exactly one key named "answer".
        Its value must be {answer_formats[answer_type]}.
        """

    if needs_retrieval:
        chunks = retriever.retrieve(query=question)
        retrieved_context = "\n\n".join(chunk.text for chunk in chunks)
    else:
        retrieved_context = ""

    if needs_reasoning:
        reasoning_instruction = """
        Identify the relevant facts, apply the required rules or calculations,
        verify the result, and then return only the structured final answer.
        """
    else:
        reasoning_instruction = ""

    prompt = f"""
    Answer the following question. Use the question, supplied context and
    retrieved context as evidence for document-specific facts.

    Question:
    {question}

    Supplied context:
    {provided_context}

    Retrieved context:
    {retrieved_context}

    Additional instruction:
    {reasoning_instruction}

    The answer type is: {answer_type}

    {output_instruction}
    """

    return call_model(prompt)
