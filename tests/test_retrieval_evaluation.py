import unittest
from datetime import datetime, timedelta, timezone

from core.retrieval_evaluation import as_timestamp, build_retrieval_report


class RetrievalEvaluationTests(unittest.TestCase):
    def test_report_tracks_repeat_frequency_and_age_distribution(self):
        now = datetime(2026, 6, 5)
        report = build_retrieval_report(
            [
                {
                    "memory_id": "fresh",
                    "created_at": (now - timedelta(days=2)).timestamp(),
                    "retrieval_count": 1,
                },
                {
                    "memory_id": "repeat",
                    "created_at": (now - timedelta(days=45)).timestamp(),
                    "retrieval_count": 7,
                },
                {
                    "memory_id": "old",
                    "created_at": (now - timedelta(days=120)).timestamp(),
                    "retrieval_count": 0,
                },
                {"memory_id": "unknown", "retrieval_count": 0},
            ],
            now=now,
            repeat_threshold=5,
        )

        self.assertEqual(report["total_memories"], 4)
        self.assertEqual(report["retrieved_memory_count"], 2)
        self.assertEqual(report["repeated_memory_count"], 1)
        self.assertEqual(report["memory_repeat_frequency"][0]["memory_id"], "repeat")
        self.assertEqual(report["memory_age_distribution"]["0_7_days"], 1)
        self.assertEqual(report["memory_age_distribution"]["31_90_days"], 1)
        self.assertEqual(report["memory_age_distribution"]["over_90_days"], 1)
        self.assertEqual(report["memory_age_distribution"]["unknown"], 1)

    def test_naive_iso_datetimes_are_treated_as_utc(self):
        now = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
        created = "2026-06-03T12:00:00"

        self.assertEqual(
            as_timestamp(created),
            datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc).timestamp(),
        )

        report = build_retrieval_report(
            [{"memory_id": "naive-iso", "created_at": created, "retrieval_count": 1}],
            now=now,
        )

        self.assertEqual(report["memory_age_distribution"]["0_7_days"], 1)


if __name__ == "__main__":
    unittest.main()
