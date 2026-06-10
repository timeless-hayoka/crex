import unittest

from core.dii_tracker import DIITracker
from core.sensor_initialization import initialize_sensors


class DIITrackerTests(unittest.TestCase):
    def test_tracker_defaults_until_heartbeat_then_wakes(self):
        tracker = DIITracker()

        self.assertEqual(tracker.get_current(), 0.5)
        self.assertFalse(tracker.is_awake())

        value = tracker.update_from_interaction("System initialization sequence complete.")

        self.assertTrue(tracker.is_awake())
        self.assertEqual(tracker.get_current(), value)
        self.assertNotEqual(tracker.get_current(), 0.5)

    def test_tracker_variance_after_distinct_interactions(self):
        """
        Verify that a DIITracker reports positive variance after two distinct interactions.
        
        Calls update_from_interaction with two different messages and asserts that tracker.variance() is greater than 0.0.
        """
        tracker = DIITracker()
        tracker.update_from_interaction("Short baseline.")
        tracker.update_from_interaction(
            "Longer reflective interaction with multiple clauses, structure, and distinct terms."
        )

        self.assertGreater(tracker.variance(), 0.0)

    def test_initialize_sensors_returns_online_status(self):
        status = initialize_sensors("System initialization sequence complete.")

        self.assertTrue(status["dii_awake"])
        self.assertGreaterEqual(status["dii_samples"], 1)
        self.assertGreater(status["dii_current"], 0.0)


if __name__ == "__main__":
    unittest.main()
