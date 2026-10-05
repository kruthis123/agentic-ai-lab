import unittest

from src.scoring import ScoreResult, score


class ScoreTests(unittest.TestCase):
    def test_text_is_trimmed_and_case_insensitive(self) -> None:
        result = score('{"answer": "  TOKYO  "}', "text", "Tokyo")

        self.assertEqual(result, ScoreResult(True, "  TOKYO  ", None))

    def test_numeric_uses_fixed_absolute_tolerance(self) -> None:
        self.assertTrue(score('{"answer": 206.009}', "numeric", 206).correct)
        self.assertFalse(score('{"answer": 206.02}', "numeric", 206).correct)

    def test_numeric_rejects_string_values(self) -> None:
        result = score('{"answer": "206"}', "numeric", 206)

        self.assertEqual(result.error_type, "type_error")

    def test_date_requires_a_valid_iso_date(self) -> None:
        self.assertTrue(
            score('{"answer": "2026-09-01"}', "date", "2026-09-01").correct
        )
        self.assertEqual(
            score('{"answer": "01-09-2026"}', "date", "2026-09-01").error_type,
            "type_error",
        )

    def test_set_comparison_ignores_order_case_and_whitespace(self) -> None:
        result = score('{"answer": [" e2", "E1 "]}', "set", ["E1", "E2"])

        self.assertTrue(result.correct)

    def test_set_comparison_rejects_missing_or_additional_items(self) -> None:
        self.assertFalse(score('{"answer": ["E2"]}', "set", ["E1", "E2"]).correct)
        self.assertFalse(
            score('{"answer": ["E1", "E2", "E3"]}', "set", ["E1", "E2"]).correct
        )

    def test_invalid_json_is_a_format_error(self) -> None:
        result = score('{"answer":', "text", "Tokyo")

        self.assertEqual(result, ScoreResult(False, None, "format_error"))

    def test_missing_answer_field_is_a_format_error(self) -> None:
        result = score('{"result": "Tokyo"}', "text", "Tokyo")

        self.assertEqual(result, ScoreResult(False, None, "format_error"))

    def test_wrong_valid_answer_is_reported(self) -> None:
        result = score('{"answer": 421}', "numeric", 420)

        self.assertEqual(result.error_type, "wrong_answer")

    def test_invalid_gold_answer_raises(self) -> None:
        with self.assertRaises(ValueError):
            score('{"answer": 420}', "numeric", [420])

    def test_unsupported_answer_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            score('{"answer": true}', "boolean", True)


if __name__ == "__main__":
    unittest.main()
