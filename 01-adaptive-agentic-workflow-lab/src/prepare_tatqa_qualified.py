"""Refine five audited questions and keep only retrieval-qualified context pairs."""

import copy
import hashlib
import json
import math
from pathlib import Path

from src.retrieval import TfidfRetriever, load_tatqa_chunks
from src.tatqa_scoring import score_tatqa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Clarify wording without changing the source annotation; replacements use source QIDs.
CLARIFICATIONS = {
    "E7-arithmetic": "What is the change in the weighted average hedged rate from 2018 to 2019, in percentage points?",
    "E21-lookup": "How many freehold investment properties acquired during the year ended 30 June 2019 did the Group obtain external valuations on?",
}
REPLACEMENTS = {
    "E16-arithmetic": "82c155de-a907-4b3f-b89f-1984bd97ddc5",
    "E25-lookup": "d023e08e-4208-4fed-b81f-665840cec045",
    "E25-arithmetic": "0d0c77cf-d32c-4b0e-8da0-dcaa3f681f59",
}
# Values and calculations reviewed against the source before any model run.
REFINED_REFERENCES = {
    "E7-arithmetic": ({"answer": 0.5, "scale": "percent"}, "2.10 - 1.60 = 0.50 percentage points"),
    "E21-lookup": ({"answer": 31, "scale": ""}, "Paragraph 2: 31 acquired freehold properties in year ended 30 June 2019"),
    "E16-arithmetic": ({"answer": 12.67, "scale": "percent"}, "(138.7 / 123.1 - 1) * 100 = 12.67% after rounding"),
    "E25-lookup": ({"answer": 3.6, "scale": "percent"}, "Bell Wireless operating revenue change, Q4 2019 versus Q4 2018, 3.6%"),
    "E25-arithmetic": ({"answer": 39.47, "scale": "percent"}, "2493 / 6316 * 100 = 39.47% after rounding; table units cancel"),
}


def main():
    input_path = DATA_DIR / "tatqa-evaluation.jsonl"
    original_rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    contexts = json.loads((DATA_DIR / "tatqa-evaluation-contexts.json").read_text())
    source = json.loads((DATA_DIR / "tatqa_dataset_dev.json").read_text())
    source_questions = {q["uid"]: q for context in source for q in context["questions"]}
    source_contexts = {context["table"]["uid"]: context for context in source}
    old_audit = json.loads((DATA_DIR / "tatqa-evaluation-audit.json").read_text())
    assert old_audit["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    audit_by_id = {row["id"]: row for row in old_audit["questions"]}
    rows = copy.deepcopy(original_rows)
    changes = []

    for row in rows:
        base_id = row["id"].rsplit("-", 1)[0]
        original_question = row["question"]
        original_qid = row["source_question_id"]
        if base_id in REPLACEMENTS:
            question = source_questions[REPLACEMENTS[base_id]]
            row["source_question_id"] = question["uid"]
            row["gold_annotation"] = question
            row["question"] = question["question"]
        if base_id in CLARIFICATIONS:
            row["question"] = CLARIFICATIONS[base_id]
        row["source_question_text"] = row["gold_annotation"]["question"]
        if base_id in REFINED_REFERENCES and row["condition"] == "supplied":
            changes.append({
                "id": base_id,
                "method": "wording_clarification" if base_id in CLARIFICATIONS else "replacement_with_source_question",
                "original_question_id": original_qid,
                "selected_question_id": row["source_question_id"],
                "original_question": original_question,
                "refined_question": row["question"],
                "source_question_text": row["source_question_text"],
                "new_gold_answer": row["gold_annotation"]["answer"],
                "new_gold_scale": row["gold_annotation"]["scale"],
            })

    assert math.isclose(round((138.7 / 123.1 - 1) * 100, 2), 12.67)
    assert math.isclose(round(2493 / 6316 * 100, 2), 39.47)
    qualified_contexts, qualified_rows, qualified_audit, excluded = [], [], [], []
    for context in contexts:
        original_context = source_contexts[context["context_id"]]
        assert context["table"] == original_context["table"]
        assert context["paragraphs"] == original_context["paragraphs"]
        retriever = TfidfRetriever(load_tatqa_chunks(context))
        context_rows = [row for row in rows if row["context_id"] == context["context_id"]]
        audited = []
        for row in context_rows:
            if row["condition"] != "retrieval":
                continue
            base_id = row["id"].rsplit("-", 1)[0]
            supplied = next(item for item in context_rows if item["id"] == base_id + "-supplied")
            assert supplied["question"] == row["question"]
            assert supplied["gold_annotation"] == row["gold_annotation"]
            assert row["provided_context"] == ""
            assert supplied["provided_context"] == context["context_text"]
            assert row["gold_annotation"] == source_questions[row["source_question_id"]]
            assert row["source_question_text"] == source_questions[row["source_question_id"]]["question"]
            previous = audit_by_id[base_id]
            reference, basis = REFINED_REFERENCES.get(base_id, (previous["reference_response"], previous["verified_basis"]))
            score = score_tatqa(json.dumps(reference), row["gold_annotation"])
            assert score["correct"], base_id
            chunk_ids = [chunk.chunk_id for chunk in retriever.retrieve(row["question"])]
            required = "paragraph-2" if base_id == "E21-lookup" else "table"
            # E25 lookup is now a precomputed percentage in the retrieved table.
            audited.append({
                "id": base_id,
                "source_question_id": row["source_question_id"],
                "question": row["question"],
                "reference_response": reference,
                "verified_basis": basis,
                "reference_exact_match": score["exact_match"],
                "reference_literal_scale_correct": score["scale_correct"],
                "required_chunk": required,
                "retrieved_chunk_ids": chunk_ids,
                "evidence_retrieved": required in chunk_ids,
                "warning": previous["warning"],
            })
        assert len(context_rows) == 4 and len(audited) == 2
        if all(item["evidence_retrieved"] for item in audited):
            qualified_contexts.append(context)
            qualified_rows.extend(context_rows)
            qualified_audit.extend(audited)
        else:
            excluded.append({
                "context_id": context["context_id"],
                "input_ids": [row["id"] for row in context_rows],
                "reason": "At least one required evidence chunk is absent from top-2 retrieval; entire context excluded to preserve four balanced conditions.",
                "question_checks": audited,
            })

    manifest = {
        "status": "Refined and retrieval-qualified; pending Task 3 protocol freeze; no model evaluation",
        "source_selection": "tatqa-evaluation-selection.json",
        "source_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "scope": "Controlled routing comparison conditional on successful retrieval, not end-to-end evaluation on unfiltered TAT-QA.",
        "filter_rule": "Apply the five approved refinements, then keep a context only when both questions' required evidence chunks appear in the unchanged TF-IDF retriever's top-2 results.",
        "sampling": "No new sampling or replacement contexts; retains a qualified subset of the original seed-42 selection.",
        "original_context_count": len(contexts),
        "selected_context_count": len(qualified_contexts),
        "input_count": len(qualified_rows),
        "inputs_per_route": len(qualified_contexts),
        "refinements": changes,
        "excluded_contexts": excluded,
        "limitations": [
            "Retrieval qualification uses source evidence annotations before model runs; gold answers and qualification metadata must never enter router/answer prompts.",
            "Results must be described as retrieval-qualified; excluded retrieval failures are outside this experiment's scope.",
            "Successful evidence retrieval does not guarantee correct interpretation or final answers.",
            "Percentage strings with source scale '' can yield correct EM with response scale percent while literal scale accuracy is false.",
            "E22 arithmetic uses positive liability magnitudes; E3 and E18 arithmetic remain deliberately easy cases.",
        ],
    }
    audit = {
        "status": "All retained questions pass evidence retrieval and reference-response scoring; no blocking question issues remain from the original audit.",
        "source_questions": len(qualified_audit),
        "variants": len(qualified_rows),
        "retrieval_passes": len(qualified_audit),
        "retrieval_misses": 0,
        "questions": qualified_audit,
    }
    for filename, value in [
        ("tatqa-evaluation-qualified-contexts.json", qualified_contexts),
        ("tatqa-evaluation-qualified-selection.json", manifest),
        ("tatqa-evaluation-qualified-audit.json", audit),
    ]:
        (DATA_DIR / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (DATA_DIR / "tatqa-evaluation-qualified.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in qualified_rows) + "\n"
    )
    print(f"Refined {len(changes)} questions; kept {len(qualified_contexts)}/{len(contexts)} contexts.")
    print(f"Saved {len(qualified_rows)} inputs, {len(qualified_contexts)} per route.")
    print(f"Retrieval passed {len(qualified_audit)}/{len(qualified_audit)} source questions. No model calls.")


if __name__ == "__main__":
    main()
