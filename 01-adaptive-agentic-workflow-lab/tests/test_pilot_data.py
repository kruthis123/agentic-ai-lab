from collections import Counter
import json
from pathlib import Path
import unittest

from src.scoring import score


LAB_ROOT = Path(__file__).resolve().parents[1]
PILOT_DATA_PATH = LAB_ROOT / "data" / "pilot-queries.jsonl"


class PilotDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.queries = [
            json.loads(line)
            for line in PILOT_DATA_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_contains_twelve_unique_queries(self) -> None:
        ids = [query["id"] for query in self.queries]

        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)

    def test_has_three_queries_for_each_expected_route(self) -> None:
        route_counts = Counter(
            (
                query["expected_route"]["needs_retrieval"],
                query["expected_route"]["needs_reasoning"],
            )
            for query in self.queries
        )

        self.assertEqual(
            route_counts,
            Counter(
                {
                    (False, False): 3,
                    (True, False): 3,
                    (False, True): 3,
                    (True, True): 3,
                }
            ),
        )

    def test_every_gold_answer_satisfies_its_scoring_contract(self) -> None:
        for query in self.queries:
            with self.subTest(query_id=query["id"]):
                response = json.dumps({"answer": query["gold_answer"]})
                result = score(response, query["answer_type"], query["gold_answer"])

                self.assertTrue(result.correct)


if __name__ == "__main__":
    unittest.main()
