import unittest

from src.scoring import score
from src.strategies import run_adaptive, run_always_complex


QUESTION = "'Employee: Maya Chen. Destination: Tokyo.' What is the destination?"
ANSWER_TYPE = "text"
GOLD_ANSWER = "Tokyo"


def print_result(result: dict) -> None:
    score_result = score(result["answer_text"], ANSWER_TYPE, GOLD_ANSWER)

    print(f"\n{result['strategy']}")
    print(f"Answer: {result['answer_text']}")
    print(
        "Route: "
        f"retrieval={result['needs_retrieval']}, "
        f"reasoning={result['needs_reasoning']}"
    )
    print(f"Correct: {score_result.correct}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Latency: {result['latency_ms']:.2f} ms")


class TestStrategies(unittest.TestCase):
    def test_one_pilot_query(self) -> None:
        adaptive_result = run_adaptive(QUESTION, ANSWER_TYPE)
        baseline_result = run_always_complex(QUESTION, ANSWER_TYPE)

        print_result(adaptive_result)
        print_result(baseline_result)

        self.assertTrue(adaptive_result["answer_text"])
        self.assertTrue(baseline_result["answer_text"])
        self.assertGreater(adaptive_result["total_tokens"], 0)
        self.assertGreater(baseline_result["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
