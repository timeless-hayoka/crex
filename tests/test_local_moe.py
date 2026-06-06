import unittest

import numpy as np

from core.dii_tracker import DIITracker
from core.local_moe import (
    LocalMoERouter,
    speculative_lookahead_tau,
    vector_from_dii_summary,
    verify_moe_pipeline,
)


class LocalMoETests(unittest.TestCase):
    def test_top_one_route_executes_one_expert(self):
        router = LocalMoERouter(num_experts=4, dimension=4, seed=7)
        result = router.route(np.array([[0.1, 0.9, -0.4, 0.3]]), k=1)
        counts = router.execution_counts()

        self.assertEqual(result.output.shape, (1, 4))
        self.assertEqual(result.active_expert_count, 1)
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(len([count for count in counts.values() if count > 0]), 1)

    def test_top_two_route_executes_two_experts(self):
        """
        Verifies that routing with k=2 activates exactly two experts and that per-expert execution counts reflect two total executions.
        
        Asserts that:
        - the router reports two active experts,
        - the sum of all execution counts equals 2,
        - exactly two experts have a positive execution count.
        """
        router = LocalMoERouter(num_experts=6, dimension=4, seed=3)
        result = router.route(np.array([[0.2, -0.1, 0.7, 0.4]]), k=2)
        counts = router.execution_counts()

        self.assertEqual(result.active_expert_count, 2)
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(len([count for count in counts.values() if count > 0]), 2)

    def test_speculative_tau_increases_with_acceptance(self):
        low = speculative_lookahead_tau(acceptance_rate=0.2, gamma=4)
        high = speculative_lookahead_tau(acceptance_rate=0.8, gamma=4)

        self.assertGreater(high, low)
        self.assertGreater(high, 3.0)

    def test_dii_summary_builds_router_vector(self):
        tracker = DIITracker()
        tracker.update_from_interaction("System initialization sequence complete.")
        vector = vector_from_dii_summary(tracker.summary())

        self.assertEqual(vector.shape, (1, 4))
        self.assertEqual(float(vector[0, 3]), 1.0)

    def test_self_check_passes(self):
        self.assertTrue(verify_moe_pipeline())


if __name__ == "__main__":
    unittest.main()
