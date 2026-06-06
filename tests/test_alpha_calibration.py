import inspect
import unittest

from core.alpha_calibration import fit_alpha
from scripts.alpha_calibration_run import calibrate_records, synthetic_trajectory


class AlphaCalibrationTests(unittest.TestCase):
    def test_fit_alpha_detects_line(self):
        records = [
            {"response_len": length, "delta_energy": 0.00002 * length}
            for length in range(100, 700, 25)
        ]

        result = fit_alpha(records, min_samples=20)

        self.assertEqual(result.verdict, "line_detected")
        self.assertAlmostEqual(result.alpha, 0.00002, places=8)
        self.assertGreater(result.correlation, 0.99)

    def test_fit_alpha_rejects_insufficient_data(self):
        result = fit_alpha([{"response_len": 100, "delta_energy": 0.01}], min_samples=20)

        self.assertIsNone(result.alpha)
        self.assertTrue(result.verdict.startswith("insufficient_data"))

    def test_synthetic_trajectory_runs_fifty_turns_with_gates(self):
        records = synthetic_trajectory(seed=7, turns=50)
        result = fit_alpha(records)

        self.assertEqual(len(records), 50)
        self.assertGreater(sum(1 for record in records if record["gate_applied"]), 0)
        self.assertEqual(result.verdict, "line_detected")

    def test_synthetic_trajectory_uses_secure_temp_log(self):
        source = inspect.getsource(synthetic_trajectory)

        self.assertIn("NamedTemporaryFile", source)
        self.assertIn("delete=False", source)
        self.assertIn("tmp.close()", source)
        self.assertNotIn("/tmp/drift_alpha_calibration.jsonl", source)

    def test_calibration_runner_keeps_fixed_sample_floor(self):
        short_result = calibrate_records(
            [{"response_len": length, "delta_energy": 0.00002 * length} for length in range(1, 5)]
        )
        enough_result = calibrate_records(
            [{"response_len": length, "delta_energy": 0.00002 * length} for length in range(100, 600, 25)]
        )

        self.assertEqual(short_result["verdict"], "insufficient_records:4/20")
        self.assertIsNone(short_result["alpha"])
        self.assertEqual(enough_result["verdict"], "line_detected")


if __name__ == "__main__":
    unittest.main()
