"""Validate our JSON response, then apply official TAT-QA scoring rules."""

import json
import math

from src.tatqa_metrics import (
    add_percent_pred,
    extract_gold_answers,
    get_answer_str,
    get_metrics,
    metric_max_over_ground_truths,
)


def _reject_non_json_number(value):
    raise ValueError(f"Invalid JSON number: {value}")


def score_tatqa(response: str, gold_annotation: dict) -> dict:
    result = {
        "correct": False,
        "parsed_answer": None,
        "parsed_scale": None,
        "exact_match": 0.0,
        "f1": 0.0,
        "scale_correct": False,
        "error_type": None,
    }
    try:
        payload = json.loads(response, parse_constant=_reject_non_json_number)
    except (ValueError, TypeError):
        return {**result, "error_type": "format_error"}

    if not isinstance(payload, dict) or set(payload) != {"answer", "scale"}:
        return {**result, "error_type": "format_error"}

    answer, scale = payload["answer"], payload["scale"]
    result.update(parsed_answer=answer, parsed_scale=scale)
    prediction = answer if isinstance(answer, list) else [answer]
    if scale not in ["", "thousand", "million", "billion", "percent"]:
        return {**result, "error_type": "type_error"}
    for value in prediction:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return {**result, "error_type": "type_error"}
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return {**result, "error_type": "type_error"}

    gold_type, gold_answer, gold_scale = extract_gold_answers(gold_annotation)
    prediction = [str(value) for value in prediction]
    predicted_strings = get_answer_str(prediction, scale)
    predicted_strings = add_percent_pred(predicted_strings, scale, prediction)
    gold_strings = get_answer_str(gold_answer, gold_scale)
    exact_match, f1 = metric_max_over_ground_truths(
        get_metrics, predicted_strings, gold_strings
    )
    if gold_type in ["arithmetic", "count"]:
        f1 = exact_match

    result.update(
        correct=exact_match == 1.0,
        exact_match=float(exact_match),
        f1=float(f1),
        scale_correct=scale == gold_scale,
        error_type=None if exact_match == 1.0 else "wrong_answer",
    )
    return result
