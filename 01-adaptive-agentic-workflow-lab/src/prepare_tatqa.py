"""Prepare five deliberately simple training contexts for development."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Context UID followed by lookup and arithmetic question UIDs.
SELECTION = [
    ("f916c7cb-7070-457f-abc0-93b9fae23127",
     "dd21282a-ea76-47eb-95a7-4c72cb6b2e84",
     "fea9fc8d-84e2-4c79-bbdf-69417d648201"),
    ("e12a0712-0fcd-475e-8549-d3691d7c01f8",
     "d13f36a4-2579-4c38-b7b1-de8a3ff01183",
     "6dd96fb3-f28e-4922-86c3-81abfe7dbaaa"),
    ("71af41cf-6910-491c-861d-c3d70a4853b7",
     "4bd8fcbd-f70f-42ad-b8d8-36cb357c2126",
     "45eef2ce-a6ee-4f79-b2d8-3b568bd7389e"),
    ("f236fdd6-92a6-4501-bd35-c28104b49dcd",
     "bbbb2d7f-b24a-40d4-9254-faf66ebd16a2",
     "d44b4eae-0d89-4279-8ede-e25e92fa3b6a"),
    ("435ff38e-82de-4b9d-9c1b-0d7ef2857e14",
     "d79b720b-d7a4-4f4b-8632-d4d53b198afb",
     "7e26db7c-93e7-4d59-a3de-9ec2ece7d04a"),
]


def main():
    training = json.loads(
        (DATA_DIR / "tatqa_dataset_train.json").read_text(encoding="utf-8")
    )
    sources = {item["table"]["uid"]: item for item in training}
    contexts = []
    variants = []

    for number, (context_id, lookup_id, arithmetic_id) in enumerate(SELECTION, 1):
        source = sources[context_id]
        questions = {q["uid"]: q for q in source["questions"]}
        table_text = "\n".join(" | ".join(row) for row in source["table"]["table"])
        full_context = "\n\n".join(
            [p["text"] for p in source["paragraphs"]] + [table_text]
        )
        contexts.append({
            "context_id": context_id,
            "source_split": "train",
            "table": source["table"],
            "paragraphs": source["paragraphs"],
            "context_text": full_context,
        })

        for kind, question_id in [("lookup", lookup_id), ("arithmetic", arithmetic_id)]:
            question = questions[question_id]
            for condition in ["supplied", "retrieval"]:
                variants.append({
                    "id": f"T{number}-{kind}-{condition}",
                    "context_id": context_id,
                    "source_question_id": question_id,
                    "question": question["question"],
                    "condition": condition,
                    "provided_context": full_context if condition == "supplied" else "",
                    "expected_route": {
                        "needs_retrieval": condition == "retrieval",
                        "needs_reasoning": kind == "arithmetic",
                    },
                    # Evaluation only: never include this in a model prompt.
                    "gold_annotation": question,
                })

    (DATA_DIR / "tatqa-development-contexts.json").write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "tatqa-development.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in variants) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(contexts)} training contexts and {len(variants)} development inputs.")


if __name__ == "__main__":
    main()
