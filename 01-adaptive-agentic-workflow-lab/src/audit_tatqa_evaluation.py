"""Save the source-grounded Task 2 audit; no model calls or selection changes."""

import hashlib
import json
import math
from pathlib import Path

from src.retrieval import TfidfRetriever, load_tatqa_chunks
from src.tatqa_scoring import score_tatqa

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Reviewed against the selected tables/paragraphs, not inferred from model answers.
# Lookup value, output scale, lookup evidence, arithmetic expression, computed value.
CHECKS = [
    (31, "million", "Deferred compensation plan assets, April 27 2018", "(29-25)/25*100", (29-25)/25*100),
    (262, "thousand", "Shares purchased, January 26-February 22 2019; shares in thousands", "3608-3380", 3608-3380),
    (2.9, "percent", "Discount rate, 2019; column units %", "2.5-2.5", 2.5-2.5),
    (195837, "thousand", "Romania net sales, 2019; paragraph 3 specifies thousands", "(14271-12953)/12953*100", (14271-12953)/12953*100),
    (1230, "thousand", "Interest cost, fiscal 2018; paragraph 6 specifies thousands", "(470-240)/240*100", (470-240)/240*100),
    (1496.5, "million", "Total sales, 2019; paragraph 2 specifies millions", "(44.1-56.7)/56.7*100", (44.1-56.7)/56.7*100),
    (-494, "million", "2019 carrying amount (494); RMB Million column", "2.10-1.60", 2.10-1.60),
    (112852, "thousand", "Operating income GAAP, 2018; paragraph 1 specifies thousands", "(237235-142105)/142105*100", (237235-142105)/142105*100),
    (13428, "million", "Revenue, July 27 2019; paragraph 2 specifies millions", "3044-2822", 3044-2822),
    (788948, "thousand", "Sales, 2019; paragraph 3 specifies thousands", "(315652-365607)/365607*100", (315652-365607)/365607*100),
    (100.1, "million", "Total liabilities, revised December 31 2019; table units millions", "100.1/545.8*100", 100.1/545.8*100),
    (41.5, "percent", "Gross margin, fiscal 2019 second quarter, 41.5%", "88081/365912*100", 88081/365912*100),
    (2027, "", "Domestic-state tax CREDIT carryforwards, expiration year", "39784+3313+15345", 39784+3313+15345),
    (708, "million", "Accounts receivable net, as reported; table units millions", "114/15824*100", 114/15824*100),
    (17116, "", "United States, customer headquarters, 2018; no table scale provided", "(1761-1429)/1429*100", (1761-1429)/1429*100),
    (2.5, "million", "State licenses, 2018; paragraph 2 specifies millions", "136.2-120.6", 136.2-120.6),
    (206.2, "", "Basic EPS total, 2019, cents per share (not millions)", "113.6/204.9*100", 113.6/204.9*100),
    (87128, "", "Total gross emissions, FY19, tonnes CO2e", "-27603-0", -27603-0),
    (255, "thousand", "Cash-settled awards, 2019; paragraph 1 specifies thousands", "(5015-5256)/5256*100", (5015-5256)/5256*100),
    (22, "million", "State income taxes net of federal benefit; January 3 2020 column", "(-9-12)/2", (-9-12)/2),
    (31, "", "Paragraph 2: 31 properties acquired in year ended June 30 2019; also 19 in 2018", "(62+73)/2", (62+73)/2),
    (27728, "million", "USD current monetary assets, December 31 2019; RMB Million column", "14732+5739 (liability magnitudes)", 14732+5739),
    (185, "million", "Gains reclassified into revenue, 2018; table units millions", "(159-219)/219*100", (159-219)/219*100),
    (3037000, "", "Total revenue, 2019; dollar value with no stated multiplier", "(3037000-22106000)/22106000*100", (3037000-22106000)/22106000*100),
    (2493, "million", "Bell Wireless Q4 2019, table value 2493; million scale is not explicit in supplied table/caption", "2407+2493", 2407+2493),
]

BLOCKERS = {
    "E7-arithmetic": "Wording asks percent change. Gold 0.5 is a percentage-point difference; relative change is 31.25%. Replace with an unambiguous source arithmetic question or explicitly clarify the derived question.",
    "E16-arithmetic": "Paragraph 4 explicitly says FCC licenses increased 15.6 million. The supplied-context variant can be answered by extraction; the arithmetic route label is not justified for this full context.",
    "E21-lookup": "Question omits year; paragraph 2 gives 31 properties in 2019 and 19 in 2018. Gold assumes 2019 and refers to newly acquired properties. Clarify scope/date or replace the question/context.",
    "E25-lookup": "Gold uses million, but the revenue table and caption do not explicitly state that scale. Other million-denominated figures do not establish this table's units. Replace with a supported percentage lookup or resolve source units explicitly.",
    "E25-arithmetic": "Sum 4900 is verified, but its million scale is not explicit in the supplied revenue table/caption. A percentage-of-total question avoids this missing scale requirement.",
}

WARNINGS = {
    "E3-arithmetic": "Zero-change calculation: equal rates make this very easy. Keep visible; an operation label does not prove extra reasoning is beneficial.",
    "E12-lookup": "Gold stores 41.5% with scale ''. The valid numeric response 41.5 with scale percent has EM=1 but literal scale_correct=False. Do not call this a unit error or override official scores.",
    "E15-lookup": "No multiplier is stated; source gold scale is ''. Return the displayed value, without assuming thousands or millions.",
    "E18-arithmetic": "Subtracting a zero baseline gives a change equal to the displayed FY19 amount; this is a weak reasoning-demand example, not an invalid gold answer.",
    "E22-arithmetic": "Gold sums positive liability magnitudes. The table shows negative signed liabilities; a signed sum would be -20471. Document this convention or use an unambiguous assets-sum question.",
}


def main():
    input_path = DATA_DIR / "tatqa-evaluation.jsonl"
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    contexts = json.loads((DATA_DIR / "tatqa-evaluation-contexts.json").read_text())
    sources = json.loads((DATA_DIR / "tatqa_dataset_dev.json").read_text())
    source_by_id = {source["table"]["uid"]: source for source in sources}
    original = {q["uid"]: q for c in sources for q in c["questions"]}
    records = []
    retrieval_misses = []

    assert len(contexts) == len(CHECKS) == 25 and len(rows) == 100
    for number, (context, check) in enumerate(zip(contexts, CHECKS), 1):
        source = source_by_id[context["context_id"]]
        assert context["table"] == source["table"] and context["paragraphs"] == source["paragraphs"]
        lookup_value, lookup_scale, lookup_evidence, expression, computed = check
        retriever = TfidfRetriever(load_tatqa_chunks(context))
        for kind in ["lookup", "arithmetic"]:
            base_id = f"E{number}-{kind}"
            supplied = next(row for row in rows if row["id"] == base_id + "-supplied")
            retrieval = next(row for row in rows if row["id"] == base_id + "-retrieval")
            gold = supplied["gold_annotation"]
            for row in [supplied, retrieval]:
                assert row["gold_annotation"] == original[row["source_question_id"]]
                assert row["question"] == original[row["source_question_id"]]["question"]
                assert row["context_id"] == context["context_id"]
                assert row["expected_route"]["needs_reasoning"] == (kind == "arithmetic")
            assert supplied["provided_context"] == context["context_text"]
            assert retrieval["provided_context"] == ""
            assert not supplied["expected_route"]["needs_retrieval"] and retrieval["expected_route"]["needs_retrieval"]
            if kind == "arithmetic":
                assert math.isclose(round(computed, 2), gold["answer"], abs_tol=1e-9)
            value = lookup_value if kind == "lookup" else gold["answer"]
            scale = lookup_scale if kind == "lookup" else gold["scale"]
            reference = {"answer": value, "scale": scale}
            score = score_tatqa(json.dumps(reference), gold)
            assert score["correct"], (base_id, score)
            chunks = retriever.retrieve(supplied["question"])
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            required_chunk = "paragraph-2" if base_id == "E21-lookup" else "table"
            evidence_retrieved = required_chunk in chunk_ids
            if not evidence_retrieved:
                retrieval_misses.append(retrieval["id"])
            records.append({
                "id": base_id,
                "source_question_id": supplied["source_question_id"],
                "question": supplied["question"],
                "gold_answer": gold["answer"],
                "gold_scale": gold["scale"],
                "status": "needs_resolution" if base_id in BLOCKERS else "pass_with_warning" if base_id in WARNINGS else "pass",
                "blocking_issue": BLOCKERS.get(base_id),
                "warning": WARNINGS.get(base_id),
                "verified_basis": lookup_evidence if kind == "lookup" else expression,
                "verified_value": value,
                "reference_response": reference,
                "reference_exact_match": score["exact_match"],
                "reference_literal_scale_correct": score["scale_correct"],
                "retrieved_chunk_ids": chunk_ids,
                "required_chunk": required_chunk,
                "evidence_retrieved": evidence_retrieved,
            })

    report = {
        "status": "Audit complete; selection needs resolution before protocol freeze",
        "review_method": "Assistant source review of all 25 selected tables/paragraph sets; literal calculations and reference-response scoring checked in code. No model evaluation calls.",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_questions_reviewed": len(records),
        "variants_checked": len(rows),
        "gold_calculations_match_source_derivations": 25,
        "reference_responses_pass_official_em": 50,
        "blocking_question_count": len(BLOCKERS),
        "blocking_variant_count": len(BLOCKERS) * 2,
        "retrieval_evidence_misses": retrieval_misses,
        "retrieval_miss_count": len(retrieval_misses),
        "selection_changed": False,
        "interpretation": [
            "Gold arithmetic matching a derivation does not resolve ambiguous wording, missing units, or a precomputed answer in the context.",
            "Retrieval misses are pipeline outcomes, not invalid dataset examples. Do not remove them to improve results.",
            "Current table chunks retain necessary headers/units when retrieved; paragraph-based E21 lookup needs paragraph 2 instead of the table.",
            "Keep expected labels distinct from empirically optimal routes; direct models may calculate, and some arithmetic is trivial.",
            "No evaluation examples, gold annotations, prompts or retriever settings were changed by this audit.",
        ],
        "questions": records,
    }
    path = DATA_DIR / "tatqa-evaluation-audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Audited {len(records)} questions / {len(rows)} variants.")
    print(f"All 25 source calculations verified; all 50 reference responses pass EM.")
    print(f"Questions needing resolution: {len(BLOCKERS)}; retrieval evidence misses: {len(retrieval_misses)}/50.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
