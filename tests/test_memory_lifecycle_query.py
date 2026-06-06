import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.memory_lifecycle_query import (
    build_lifecycle_report,
    load_json_metadatas,
)


class MemoryLifecycleQueryTests(unittest.TestCase):
    def test_report_counts_stale_missing_and_poor_session_memory(self):
        now = datetime(2026, 6, 5)
        recent = (now - timedelta(days=2)).timestamp()
        old = (now - timedelta(days=45)).timestamp()

        report = build_lifecycle_report(
            [
                {"memory_id": "recent", "last_accessed": recent, "retrieval_count": 3},
                {"memory_id": "stale", "last_accessed": old, "retrieval_count": 1},
                {"memory_id": "missing", "retrieval_count": 0},
                {
                    "memory_id": "poor",
                    "last_accessed": recent,
                    "retrieval_count": 8,
                    "poor_session_retrievals": 8,
                },
            ],
            now=now,
            stale_days=30,
            frequent_threshold=5,
        )

        self.assertEqual(report["total_memories"], 4)
        self.assertEqual(report["stale_count"], 1)
        self.assertEqual(report["missing_last_accessed_count"], 1)
        self.assertEqual(report["frequent_only_poor_session_count"], 1)
        self.assertEqual(report["stale_ids"], ["stale"])
        self.assertEqual(report["missing_last_accessed_ids"], ["missing"])

    def test_naive_iso_last_accessed_is_treated_as_utc(self):
        report = build_lifecycle_report(
            [
                {
                    "memory_id": "recent-naive",
                    "last_accessed": "2026-06-03T12:00:00",
                    "retrieval_count": 1,
                },
                {
                    "memory_id": "stale-aware",
                    "last_accessed": "2026-04-01T12:00:00+00:00",
                    "retrieval_count": 1,
                },
            ],
            now=datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
            stale_days=30,
        )

        self.assertEqual(report["stale_ids"], ["stale-aware"])
        self.assertEqual(report["missing_last_accessed_count"], 0)

    def test_load_json_accepts_chroma_style_metadatas_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = Path(tmp_dir) / "metadatas.json"
            fixture.write_text('{"metadatas": [{"memory_id": "a"}]}', encoding="utf-8")

            metadatas = load_json_metadatas(fixture)

        self.assertEqual(metadatas, [{"memory_id": "a"}])


if __name__ == "__main__":
    unittest.main()
