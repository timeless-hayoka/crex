import tempfile
import unittest
from pathlib import Path

from core.roi_dashboard import build_roi_dashboard, load_jsonl_records


class ROIDashboardTests(unittest.TestCase):
    def test_dashboard_aggregates_cost_benefit_and_roi(self):
        rows = build_roi_dashboard(
            [
                {
                    "subsystems": {
                        "DMU": {"cost_ms": 20, "benefit": 0.8},
                        "DII": {"cost_ms": 1},
                    }
                },
                {
                    "subsystem_costs_ms": {"DMU": 30, "PEDI": 3},
                    "subsystem_benefits": {"DMU": 0.6, "PEDI": 0.9},
                },
            ]
        )
        by_name = {row["subsystem"]: row for row in rows}

        self.assertEqual(by_name["DMU"]["samples"], 2)
        self.assertEqual(by_name["DMU"]["mean_cost_ms"], 25.0)
        self.assertEqual(by_name["DMU"]["mean_benefit"], 0.7)
        self.assertEqual(by_name["DMU"]["benefit_label"], "high")
        self.assertEqual(by_name["DII"]["benefit_label"], "unknown")
        self.assertGreater(by_name["PEDI"]["roi_per_ms"], by_name["DMU"]["roi_per_ms"])

    def test_load_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            path.write_text('{"subsystem": "DMU"}\n{"subsystem": "PEDI"}\n', encoding="utf-8")

            records = load_jsonl_records(path)

        self.assertEqual([record["subsystem"] for record in records], ["DMU", "PEDI"])


if __name__ == "__main__":
    unittest.main()
