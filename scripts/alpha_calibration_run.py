"""Run a 50-turn alpha calibration without recovery injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.alpha_calibration import fit_alpha  # noqa: E402
from core.cognitive_governor import CognitiveGovernor  # noqa: E402
from core.sensor_initialization import initialize_sensors  # noqa: E402
from core.roi_dashboard import load_jsonl_records  # noqa: E402


def synthetic_trajectory(seed: int = 42, turns: int = 50) -> list[dict[str, object]]:
    """Generate a gated calibration trajectory with no beta/recovery term."""

    if turns < 20:
        raise ValueError("turns must be at least 20 for calibration")

    rng = np.random.default_rng(seed)
    governor = CognitiveGovernor(calibration_log=Path("/tmp/drift_alpha_calibration.jsonl"))
    energy = 0.82
    true_alpha = 0.000025
    records = []

    for turn in range(1, turns + 1):
        if turn > 20:
            energy = min(energy, 0.48 if turn % 3 else 0.24)
        gate = governor.apply(energy, turn=turn)
        requested_len = int(rng.integers(80, 1300))
        response_len = min(requested_len, gate.max_tokens)
        drain = max(0.0, true_alpha * response_len + float(rng.normal(0.0, 0.00025)))
        prev_energy = energy
        energy = max(0.02, energy - drain)
        record = governor.record_turn(
            prev_energy=prev_energy,
            new_energy=energy,
            response_len=response_len,
            gate_result=gate,
        )
        records.append(
            {
                "turn": record.turn,
                "prev_energy": record.prev_energy,
                "new_energy": record.new_energy,
                "delta_energy": record.delta_energy,
                "response_len": record.response_len,
                "energy_mode": record.energy_mode,
                "max_tokens": record.max_tokens,
                "gate_applied": record.gate_applied,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", type=Path, help="Optional real trajectory records")
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSONL output path for synthetic records")
    args = parser.parse_args()

    initialize_sensors()

    if args.trajectory_jsonl:
        records = load_jsonl_records(args.trajectory_jsonl)
    else:
        records = synthetic_trajectory(seed=args.seed, turns=args.turns)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )

    result = fit_alpha(records, min_samples=min(20, len(records))).to_dict()
    print("=" * 72)
    print("ALPHA CALIBRATION: delta_energy = alpha * response_len + epsilon")
    print("=" * 72)
    print(f"samples: {result['samples']}")
    print(f"alpha: {result['alpha']}")
    print(f"correlation: {result['correlation']}")
    print(f"r_squared: {result['r_squared']}")
    print(f"verdict: {result['verdict']}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
