"""Deterministic scoring for structured experiment responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from typing import Any


SUPPORTED_ANSWER_TYPES = {"text", "numeric", "date", "set"}


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    parsed_answer: Any | None
    error_type: str | None


def score(response: str, answer_type: str, gold_answer: Any) -> ScoreResult:
    """Score a JSON response containing one ``answer`` field.

    Invalid JSON or a missing answer field is a format error. A valid answer
    with the wrong JSON type is a type error. All other mismatches are wrong
    answers. Invalid gold values raise because they indicate a dataset defect.
    """

    normalized_type = answer_type.strip().lower()
    if normalized_type not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(f"Unsupported answer type: {answer_type!r}")

    try:
        normalized_gold = _normalize(gold_answer, normalized_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid gold answer for type {normalized_type!r}: {gold_answer!r}"
        ) from exc

    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return ScoreResult(False, None, "format_error")

    if not isinstance(payload, dict) or "answer" not in payload:
        return ScoreResult(False, None, "format_error")

    parsed_answer = payload["answer"]
    try:
        normalized_answer = _normalize(parsed_answer, normalized_type)
    except (TypeError, ValueError):
        return ScoreResult(False, parsed_answer, "type_error")

    if normalized_type == "numeric":
        correct = math.isclose(
            normalized_answer,
            normalized_gold,
            rel_tol=0.0,
            abs_tol=0.01,
        )
    else:
        correct = normalized_answer == normalized_gold

    return ScoreResult(
        correct=correct,
        parsed_answer=parsed_answer,
        error_type=None if correct else "wrong_answer",
    )


def _normalize(value: Any, answer_type: str) -> Any:
    if answer_type == "text":
        if not isinstance(value, str):
            raise TypeError("Text answers must be JSON strings")
        return value.strip().casefold()

    if answer_type == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Numeric answers must be JSON numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Numeric answers must be finite")
        return number

    if answer_type == "date":
        if not isinstance(value, str):
            raise TypeError("Date answers must be ISO date strings")
        return date.fromisoformat(value.strip()).isoformat()

    if answer_type == "set":
        if not isinstance(value, list):
            raise TypeError("Set answers must be JSON arrays")
        return frozenset(_normalize_set_item(item) for item in value)

    raise ValueError(f"Unsupported answer type: {answer_type!r}")


def _normalize_set_item(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str):
        return value.strip().casefold()
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise TypeError("Set items must be JSON scalar values")
