import unittest

from core.failure_taxonomy import (
    FailureType,
    classify_failure,
    summarize_failure_types,
    with_failure_type,
)


class FailureTaxonomyTests(unittest.TestCase):
    def test_classifies_observed_raw_failures(self):
        self.assertEqual(
            classify_failure("Coordination plugin instance not found"),
            FailureType.SEAM_NOT_FOUND.value,
        )
        self.assertEqual(
            classify_failure("'Shadow' object has no attribute 'radar'"),
            FailureType.SEAM_NOT_FOUND.value,
        )
        self.assertEqual(
            classify_failure("PEDI anchor division by zero"),
            FailureType.DIVISION_BY_ZERO.value,
        )

    def test_classifies_latency_and_requested_failure_types(self):
        self.assertEqual(classify_failure(latency_seconds=12.0), FailureType.TIMEOUT.value)
        self.assertEqual(classify_failure("Gemini quota exceeded"), "quota_exceeded")
        self.assertEqual(classify_failure("math hallucination in arithmetic"), "math_hallucination")

    def test_enriches_and_summarizes_turn_metadata(self):
        enriched = with_failure_type(
            {"turn": 4},
            raw_text="DII locked at 0.5 with zero variance",
        )
        counts = summarize_failure_types(
            [
                enriched,
                {"failure_type": "timeout"},
                {"failure_type": None},
                {"failure_type": "not-a-real-type"},
            ]
        )

        self.assertEqual(enriched["failure_type"], "sensor_stuck")
        self.assertEqual(counts["sensor_stuck"], 1)
        self.assertEqual(counts["timeout"], 1)
        self.assertEqual(counts["unknown"], 1)

    def test_unknown_explicit_falls_back_to_inferred_failure(self):
        enriched = with_failure_type(
            {"failure_type": "not-a-real-type"},
            raw_text="Coordination plugin instance not found",
        )

        self.assertEqual(enriched["failure_type"], FailureType.SEAM_NOT_FOUND.value)

    def test_valid_explicit_failure_type_wins_over_inferred_failure(self):
        enriched = with_failure_type(
            {"failure_type": "timeout"},
            raw_text="Coordination plugin instance not found",
        )

        self.assertEqual(enriched["failure_type"], FailureType.TIMEOUT.value)


if __name__ == "__main__":
    unittest.main()
