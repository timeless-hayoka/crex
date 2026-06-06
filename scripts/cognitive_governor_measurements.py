"""Measurement simulation for the DRIFT cognitive governor."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cognitive_governor import CognitiveGovernor  # noqa: E402


def run_measurement(seed: int = 42, turns: int = 80) -> dict:
    """Run a synthetic closed-loop check with known response-length coupling."""

    if turns <= 30:
        raise ValueError("turns must be at least 31 to include calibration data")

    rng = np.random.default_rng(seed)
    true_alpha = 0.000025
    noise_sigma = 0.00025

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "governor_measurements.jsonl"
        governor = CognitiveGovernor(calibration_log=log_path)

        # Calibration block: keep energy normal so no token gate is applied.
        for turn in range(1, 31):
            energy = 0.82
            requested_len = int(rng.integers(250, 1200))
            gate = governor.apply(energy, turn=turn)
            response_len = min(requested_len, gate.max_tokens)
            drain = true_alpha * response_len + float(rng.normal(0.0, noise_sigma))
            governor.record_turn(
                energy,
                max(0.0, energy - drain),
                response_len=response_len,
                gate_result=gate,
            )

        fitted_alpha = governor.calibrate_alpha(min_samples=20)

        # Stress block: sampled energy falls into LOW_POWER/CRITICAL and token gates bind.
        ungated_counterfactual = []
        for turn in range(31, turns + 1):
            energy = 0.24 if turn % 5 else 0.12
            requested_len = int(rng.integers(600, 1300))
            gate = governor.apply(energy, turn=turn)
            response_len = min(requested_len, gate.max_tokens)
            gated_drain = true_alpha * response_len + float(rng.normal(0.0, noise_sigma))
            ungated_counterfactual.append(true_alpha * min(requested_len, 1000))
            governor.record_turn(
                energy,
                max(0.0, energy - gated_drain),
                response_len=response_len,
                gate_result=gate,
            )

        measurements = governor.measurements().to_dict()
        measurements["true_alpha"] = true_alpha
        measurements["fitted_alpha_error"] = (
            abs(fitted_alpha - true_alpha) if fitted_alpha is not None else None
        )
        measurements["counterfactual_ungated_stress_drain"] = round(
            float(np.mean(ungated_counterfactual)), 8
        )
        if measurements["mean_gated_drain"] is not None:
            measurements["counterfactual_gate_savings"] = round(
                measurements["counterfactual_ungated_stress_drain"]
                - measurements["mean_gated_drain"],
                8,
            )
        else:
            measurements["counterfactual_gate_savings"] = None

        return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=80)
    args = parser.parse_args()

    if args.turns <= 30:
        parser.error("turns must be at least 31 to include calibration data")

    measurements = run_measurement(seed=args.seed, turns=args.turns)

    print("=" * 72)
    print("COGNITIVE GOVERNOR MEASUREMENTS")
    print("=" * 72)
    print(f"turns: {measurements['turns']}")
    print(f"mode_distribution: {measurements['mode_distribution']}")
    print(
        "response_len mean gated/ungated: "
        f"{measurements['mean_gated_response_len']} / {measurements['mean_ungated_response_len']}"
    )
    print(
        "drain mean gated/ungated: "
        f"{measurements['mean_gated_drain']} / {measurements['mean_ungated_drain']}"
    )
    print(f"estimated_gate_savings: {measurements['estimated_gate_savings']}")
    print(f"counterfactual_gate_savings: {measurements['counterfactual_gate_savings']}")
    print(f"intervention_failures: {measurements['intervention_failures']}")
    print(f"fitted_alpha: {measurements['fitted_alpha']}")
    print(f"fitted_alpha_error: {measurements['fitted_alpha_error']}")
    print("\nJSON")
    print(json.dumps(measurements, indent=2, sort_keys=True))
    print("=" * 72)


if __name__ == "__main__":
    main()
