"""Select 25 held-out contexts with seed 42; generate 100 evaluation inputs."""

import hashlib
import json
import math
import random
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 42
CONTEXT_COUNT = 25
SCALES = ["", "thousand", "million", "billion", "percent"]


def is_numeric(value):
    if isinstance(value, bool):
        return False
    text = str(value).strip().replace(",", "").strip("$€£¥% ")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def main():
    source_path = DATA_DIR / "tatqa_dataset_dev.json"
    sources = json.loads(source_path.read_text())
    development = json.loads((DATA_DIR / "tatqa-development-contexts.json").read_text())
    excluded_ids = {context["context_id"] for context in development}
    excluded_tables = {json.dumps(context["table"]["table"]) for context in development}
    candidates = []

    # Eligibility depends only on source annotations, never on model outputs.
    for source in sorted(sources, key=lambda item: item["table"]["uid"]):
        if source["table"]["uid"] in excluded_ids or json.dumps(source["table"]["table"]) in excluded_tables:
            continue
        lookups = []
        arithmetic = []
        for question in sorted(source["questions"], key=lambda item: item["uid"]):
            if question["scale"] not in SCALES:
                continue
            answer = question["answer"]
            if (
                question["answer_type"] == "span"
                and isinstance(answer, list) and len(answer) == 1
                and is_numeric(answer[0])
                and not question.get("derivation", "").strip()
            ):
                lookups.append(question)
            elif (
                question["answer_type"] == "arithmetic"
                and isinstance(answer, (int, float)) and is_numeric(answer)
                and question.get("derivation", "").strip()
            ):
                arithmetic.append(question)
        if lookups and arithmetic:
            candidates.append((source, lookups, arithmetic))

    if len(candidates) < CONTEXT_COUNT:
        raise ValueError(f"Only {len(candidates)} contexts meet the eligibility rules")
    rng = random.Random(SEED)
    selected = rng.sample(candidates, CONTEXT_COUNT)
    contexts = []
    variants = []
    selection = []

    for number, (source, lookups, arithmetic) in enumerate(selected, 1):
        lookup = rng.choice(lookups)
        calculation = rng.choice(arithmetic)
        context_id = source["table"]["uid"]
        table_text = "\n".join(" | ".join(row) for row in source["table"]["table"])
        full_context = "\n\n".join([p["text"] for p in source["paragraphs"]] + [table_text])
        contexts.append({
            "context_id": context_id,
            "source_split": "dev",
            "table": source["table"],
            "paragraphs": source["paragraphs"],
            "context_text": full_context,
        })
        selection.append({
            "id": f"E{number}",
            "context_id": context_id,
            "lookup_question_id": lookup["uid"],
            "arithmetic_question_id": calculation["uid"],
        })
        for kind, question in [("lookup", lookup), ("arithmetic", calculation)]:
            for condition in ["supplied", "retrieval"]:
                variants.append({
                    "id": f"E{number}-{kind}-{condition}",
                    "context_id": context_id,
                    "source_question_id": question["uid"],
                    "question": question["question"],
                    "condition": condition,
                    "provided_context": full_context if condition == "supplied" else "",
                    "expected_route": {
                        "needs_retrieval": condition == "retrieval",
                        "needs_reasoning": kind == "arithmetic",
                    },
                    # Evaluation metadata only; never send these annotations to a model.
                    "gold_annotation": question,
                })

    manifest = {
        "status": "Selected; pending Task 2 audit and protocol freeze; no model runs",
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "seed": SEED,
        "source_context_count": len(sources),
        "eligible_context_count": len(candidates),
        "selected_context_count": len(contexts),
        "input_count": len(variants),
        "eligibility_rules": [
            "Exclude the development context IDs and identical development tables.",
            "Lookup: span answer, one finite numeric value, empty derivation.",
            "Arithmetic: finite numeric int/float answer and nonempty derivation.",
            "Both questions must use one of the five supported answer scales.",
            "A context must contain at least one eligible question of each kind.",
        ],
        "sampling_rule": "Sort contexts/questions by UID; sample contexts without replacement using seed 42; then randomly choose one lookup and one arithmetic question per selected context using the same RNG.",
        "limitations": [
            "This is a numeric lookup/arithmetic subset, not the full TAT-QA benchmark.",
            "100 inputs come from 50 questions in 25 contexts, not 100 independent source questions.",
            "Unseen in this experiment does not mean unseen during model pretraining.",
            "Empty derivation is a lookup-label heuristic; Task 2 must verify the required operation, evidence and scoring.",
        ],
        "selection": selection,
    }
    for filename, value in [
        ("tatqa-evaluation-contexts.json", contexts),
        ("tatqa-evaluation-selection.json", manifest),
    ]:
        (DATA_DIR / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (DATA_DIR / "tatqa-evaluation.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in variants) + "\n"
    )
    print(f"Eligible contexts: {len(candidates)}/{len(sources)}")
    print(f"Saved {len(contexts)} contexts and {len(variants)} inputs with seed {SEED}.")
    print("No model calls. Selection still requires the Task 2 audit.")


if __name__ == "__main__":
    main()
