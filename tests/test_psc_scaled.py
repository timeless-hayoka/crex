import unittest

import numpy as np

from psc_scaled import (
    PSCBatchEngine,
    _batch_chaos_score,
    _batch_residual_confidence,
    _dynamic_n_steps,
    benchmark_scale,
)


class PSCScaledTests(unittest.TestCase):
    def test_chaos_score_is_continuous_and_separates_noise(self):
        rng = np.random.default_rng(7)
        smooth = np.linspace(0.85, 0.25, 32) + rng.normal(0.0, 0.01, 32)
        noisy = rng.normal(0.65, 0.08, 32)
        history = np.column_stack([smooth, noisy])

        chaos = _batch_chaos_score(history)

        self.assertEqual(chaos.shape, (2,))
        self.assertGreater(chaos[0], 0.0)
        self.assertLess(chaos[0], 1.0)
        self.assertGreater(chaos[1], chaos[0] + 0.20)

    def test_dynamic_steps_shorten_as_chaos_increases(self):
        steps = _dynamic_n_steps(np.array([0.0, 0.35, 0.85, 1.0]))

        self.assertGreaterEqual(steps[0], steps[1])
        self.assertGreaterEqual(steps[1], steps[2])
        self.assertEqual(int(steps[0]), 10)
        self.assertEqual(int(steps[-1]), 4)

    def test_residual_confidence_penalizes_noisy_trajectory(self):
        rng = np.random.default_rng(11)
        smooth = np.linspace(0.9, 0.3, 24) + rng.normal(0.0, 0.008, 24)
        noisy = rng.normal(0.62, 0.09, 24)
        history = np.column_stack([smooth, noisy])
        chaos = _batch_chaos_score(history)
        steps = _dynamic_n_steps(chaos)
        confidence = _batch_residual_confidence(
            history,
            n_steps=steps,
            polarity=np.array([1, 1]),
            chaos_scores=chaos,
        )

        self.assertGreater(confidence[0], confidence[1] + 0.20)

    def test_engine_alerts_before_projected_focus_crisis(self):
        engine = PSCBatchEngine(["focus"], policy="SECURITY")
        data = np.linspace(0.85, 0.10, 60)
        first_alert = None

        for cycle, value in enumerate(data):
            engine.push_state({"focus": float(value)})
            result = engine.run()
            if result is not None and result.alerted[0]:
                first_alert = cycle
                break

        crisis = int(np.argmax(data <= 0.25))
        self.assertIsNotNone(first_alert)
        self.assertLess(first_alert, crisis)

    def test_engine_suppresses_stable_noisy_false_alerts(self):
        rng = np.random.default_rng(13)
        engine = PSCBatchEngine(["focus"], policy="SECURITY")
        alerts = 0

        for value in np.clip(rng.normal(0.65, 0.06, 90), 0.0, 1.0):
            engine.push_state({"focus": float(value)})
            result = engine.run()
            if result is not None:
                alerts += int(result.alerted[0])

        self.assertEqual(alerts, 0)

    def test_benchmark_scale_shape(self):
        bench = benchmark_scale([1, 4], n_cycles=12)

        self.assertEqual(set(bench), {1, 4})
        self.assertGreater(bench[1]["mean_us"], 0.0)
        self.assertGreater(bench[4]["cycles_per_sec"], 0.0)
        self.assertGreater(bench[4]["memory_bytes"], bench[1]["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
