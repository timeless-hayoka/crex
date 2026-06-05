import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.cognitive_governor import CognitiveGovernor


class CognitiveGovernorTests(unittest.TestCase):
    def test_energy_thresholds_and_invalid_sensor_hold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gov = CognitiveGovernor(calibration_log=Path(tmp_dir) / "calibration.jsonl")

            normal = gov.apply(1.0, turn=1)
            low = gov.apply(0.299, turn=2)
            critical = gov.apply(0.149, turn=3)
            hold = gov.apply(float("nan"), turn=4)

        self.assertEqual(normal.energy_mode, "NORMAL")
        self.assertEqual(normal.max_tokens, 1000)
        self.assertFalse(normal.gate_applied)
        self.assertEqual(low.energy_mode, "LOW_POWER")
        self.assertEqual(low.max_tokens, 400)
        self.assertIn("TOKEN_GATE", low.interventions)
        self.assertEqual(critical.energy_mode, "CRITICAL")
        self.assertEqual(critical.max_tokens, 150)
        self.assertEqual(hold.energy_mode, "HOLD")
        self.assertEqual(hold.max_tokens, 80)
        self.assertIn("ENERGY_SENSOR_INVALID", hold.interventions)

    def test_turn_record_log_and_measurements_track_gate_effect(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "governor.jsonl"
            gov = CognitiveGovernor(calibration_log=log_path)

            ungated = gov.apply(0.80, turn=1)
            gated = gov.apply(0.20, turn=2)
            gov.record_turn(0.80, 0.76, response_len=900, gate_result=ungated)
            gov.record_turn(0.20, 0.19, response_len=300, gate_result=gated)

            measurements = gov.measurements()
            lines = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(measurements.turns, 2)
        self.assertEqual(measurements.gated_turns, 1)
        self.assertEqual(measurements.ungated_turns, 1)
        self.assertGreater(measurements.estimated_gate_savings, 0.0)
        self.assertEqual(measurements.mode_distribution["NORMAL"], 1)
        self.assertEqual(measurements.mode_distribution["LOW_POWER"], 1)
        self.assertEqual([line["event"] for line in lines], ["TURN", "TURN"])
        self.assertEqual(lines[1]["max_tokens"], 400)

    def test_calibration_and_intervention_failure_measurement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gov = CognitiveGovernor(calibration_log=Path(tmp_dir) / "calibration.jsonl")
            rng = np.random.default_rng(9)

            for turn, response_len in enumerate(rng.integers(200, 1000, size=30), start=1):
                gate = gov.apply(0.75, turn=turn)
                drain = 0.00003 * int(response_len)
                gov.record_turn(
                    prev_energy=0.75,
                    new_energy=0.75 - drain,
                    response_len=int(response_len),
                    gate_result=gate,
                )

            alpha = gov.calibrate_alpha()
            bad_gate = gov.apply(0.14, turn=31)
            gov.record_turn(0.14, 0.10, response_len=100, gate_result=bad_gate)

            measurements = gov.measurements()

        self.assertIsNotNone(alpha)
        self.assertAlmostEqual(alpha, 0.00003, places=8)
        self.assertEqual(measurements.intervention_failures, 1)
        self.assertEqual(measurements.fitted_alpha, alpha)


if __name__ == "__main__":
    unittest.main()
