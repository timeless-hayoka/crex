"""Verify the local sparse MoE router and speculative lookahead math."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dii_tracker import DIITracker  # noqa: E402
from core.local_moe import (  # noqa: E402
    LocalMoERouter,
    speculative_lookahead_tau,
    vector_from_dii_summary,
    verify_moe_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--acceptance-rate", type=float, default=0.8)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--use-dii-vector", action="store_true")
    args = parser.parse_args()

    if args.experts == 4 and args.k == 1 and not args.use_dii_vector:
        ok = verify_moe_pipeline()
        if not ok:
            raise SystemExit(1)

    router = LocalMoERouter(num_experts=args.experts, dimension=4, seed=args.seed)
    if args.use_dii_vector:
        tracker = DIITracker()
        tracker.update_from_interaction("System initialization sequence complete.")
        tracker.update_from_interaction("Technical local routing prompt with security constraints.")
        vector = vector_from_dii_summary(tracker.summary())
    else:
        vector = np.array([[0.1, 0.9, -0.4, 0.3]])

    result = router.route(vector, k=args.k)
    report = result.to_dict()
    report["execution_counts"] = router.execution_counts()
    report["speculative_tau"] = speculative_lookahead_tau(args.acceptance_rate, args.gamma)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
