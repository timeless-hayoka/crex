"""Energy-coupled behavior control for DRIFT.

The governor converts energy from passive telemetry into an active generation
constraint. Each decision returns a token budget and an intervention audit trail
that can be logged with the turn.
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


ENERGY_MODES: Dict[str, Dict[str, object]] = {
    "NORMAL": {
        "max_tokens": 1000,
        "description": "Full cognitive capacity",
    },
    "LOW_POWER": {
        "max_tokens": 400,
        "description": "Conserving; longer responses deferred",
    },
    "CRITICAL": {
        "max_tokens": 150,
        "description": "Minimal output; system recovering",
    },
    "HOLD": {
        "max_tokens": 80,
        "description": "Sensor invalid; respond but do not act on state",
    },
}

ENERGY_THRESHOLDS = {
    "CRITICAL": (0.00, 0.15),
    "LOW_POWER": (0.15, 0.30),
    "NORMAL": (0.30, 1.00),
}


@dataclass(frozen=True)
class GateResult:
    """Output of a single governor decision."""

    max_tokens: int
    energy_mode: str
    mode_entered: bool
    interventions: List[str]
    gate_applied: bool
    energy_snapshot: float
    turn: int


@dataclass(frozen=True)
class TurnRecord:
    """One turn of causal data for alpha calibration and gate measurement."""

    turn: int
    prev_energy: float
    new_energy: float
    delta_energy: float
    response_len: int
    gate_applied: bool
    energy_mode: str
    interventions: List[str]
    max_tokens: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GovernorMeasurements:
    """Aggregated measurements proving whether interventions changed behavior."""

    turns: int
    gated_turns: int
    ungated_turns: int
    mean_drain_per_turn: float
    mean_gated_drain: Optional[float]
    mean_ungated_drain: Optional[float]
    mean_response_len: float
    mean_gated_response_len: Optional[float]
    mean_ungated_response_len: Optional[float]
    estimated_gate_savings: Optional[float]
    intervention_failures: int
    mode_distribution: Dict[str, int]
    fitted_alpha: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CognitiveGovernor:
    """Energy-coupled behavioral control.

    The governor is thread-safe around mutable mode and measurement state. It
    does not mutate the agent state; callers pass the current energy snapshot
    into :meth:`apply` before inference and record the observed post-turn energy
    with :meth:`record_turn` after homeostasis updates.
    """

    def __init__(
        self,
        calibration_log: str | Path = "logs/governor_calibration.jsonl",
        energy_min: float = 0.10,
    ) -> None:
        self._lock = threading.Lock()
        self._current_mode = "NORMAL"
        self._turn_records: List[TurnRecord] = []
        self._calibration_log = Path(calibration_log)
        self._calibration_log.parent.mkdir(parents=True, exist_ok=True)
        self.energy_min = float(energy_min)
        self._fitted_alpha: Optional[float] = None
        self._intervention_failures = 0

    def apply(self, current_energy: float, turn: int) -> GateResult:
        """Evaluate energy state and return behavioral constraints.

        Call this before inference. Pass ``GateResult.max_tokens`` to the
        generation API and persist ``GateResult.interventions`` with turn logs.
        """

        with self._lock:
            interventions: List[str] = []
            new_mode = self._compute_mode(current_energy)

            if new_mode == "HOLD":
                interventions.append("ENERGY_SENSOR_INVALID")
            elif current_energy <= self.energy_min:
                interventions.append("ENERGY_MIN_REACHED")

            mode_entered = new_mode != self._current_mode
            if mode_entered:
                self._current_mode = new_mode
                interventions.append(f"MODE_ENTERED_{new_mode}")

            unconstrained = int(ENERGY_MODES["NORMAL"]["max_tokens"])
            max_tokens = int(ENERGY_MODES[new_mode]["max_tokens"])
            gate_applied = max_tokens < unconstrained

            if gate_applied:
                interventions.append("TOKEN_GATE")

            return GateResult(
                max_tokens=max_tokens,
                energy_mode=new_mode,
                mode_entered=mode_entered,
                interventions=interventions,
                gate_applied=gate_applied,
                energy_snapshot=float(current_energy),
                turn=int(turn),
            )

    def record_turn(
        self,
        prev_energy: float,
        new_energy: float,
        response_len: int,
        gate_result: Optional[GateResult] = None,
    ) -> TurnRecord:
        """Append causal event data for calibration and intervention checks."""

        delta = float(prev_energy) - float(new_energy)
        record = TurnRecord(
            turn=gate_result.turn if gate_result else 0,
            prev_energy=float(prev_energy),
            new_energy=float(new_energy),
            delta_energy=delta,
            response_len=max(0, int(response_len)),
            gate_applied=gate_result.gate_applied if gate_result else False,
            energy_mode=gate_result.energy_mode if gate_result else "UNKNOWN",
            interventions=list(gate_result.interventions) if gate_result else [],
            max_tokens=gate_result.max_tokens if gate_result else 0,
        )

        with self._lock:
            self._turn_records.append(record)
            self._append_calibration_log(record)
            self._verify_intervention(record)

        return record

    def calibrate_alpha(self, min_samples: int = 20) -> Optional[float]:
        """Fit ``energy_drain = alpha * response_len`` from ungated turns."""

        with self._lock:
            records = [r for r in self._turn_records if not r.gate_applied]

        if len(records) < min_samples:
            print(f"Insufficient data: {len(records)} ungated turns (need {min_samples})")
            return None

        drains = np.asarray([r.delta_energy for r in records], dtype=np.float64)
        lengths = np.asarray([r.response_len for r in records], dtype=np.float64)

        if lengths.std() < 1.0:
            print("No variance in response_len; cannot fit alpha. Run diverse prompts.")
            return None

        denom = float(np.dot(lengths, lengths))
        if denom <= 0.0:
            print("No positive response lengths; cannot fit alpha.")
            return None

        alpha = float(np.dot(lengths, drains) / denom)
        corr = float(np.corrcoef(lengths, drains)[0, 1])
        if not math.isfinite(corr):
            corr = 0.0

        print("\n=== alpha CALIBRATION RESULT ===")
        print(f"  n = {len(records)} ungated turns")
        print(f"  fitted alpha = {alpha:.8f}")
        print(f"  corr(response_len, drain) = {corr:.3f}")
        print(f"  interpretation: each output character costs {alpha:.8f} energy units")
        if abs(corr) < 0.3:
            print("  WARNING: weak correlation; energy may not be coupled to response length yet.")
        elif alpha < 0:
            print("  WARNING: negative alpha; check the energy sign convention.")
        else:
            print(f"  STATUS: calibration OK; update homeostasis effort cost with alpha={alpha:.8f}")

        with self._lock:
            self._fitted_alpha = alpha
            self._append_event(
                {
                    "event": "ALPHA_CALIBRATED",
                    "samples": len(records),
                    "alpha": round(alpha, 10),
                    "correlation": round(corr, 6),
                }
            )
        return alpha

    def measurements(self) -> GovernorMeasurements:
        """Return closed-loop measurements for gate impact and drift analysis."""

        with self._lock:
            records = list(self._turn_records)
            fitted_alpha = self._fitted_alpha
            intervention_failures = self._intervention_failures

        if not records:
            return GovernorMeasurements(
                turns=0,
                gated_turns=0,
                ungated_turns=0,
                mean_drain_per_turn=0.0,
                mean_gated_drain=None,
                mean_ungated_drain=None,
                mean_response_len=0.0,
                mean_gated_response_len=None,
                mean_ungated_response_len=None,
                estimated_gate_savings=None,
                intervention_failures=intervention_failures,
                mode_distribution={},
                fitted_alpha=fitted_alpha,
            )

        gated = [r for r in records if r.gate_applied]
        ungated = [r for r in records if not r.gate_applied]
        modes: Dict[str, int] = {}
        for record in records:
            modes[record.energy_mode] = modes.get(record.energy_mode, 0) + 1

        mean_gated_drain = self._mean([r.delta_energy for r in gated])
        mean_ungated_drain = self._mean([r.delta_energy for r in ungated])
        estimated_savings = (
            round(mean_ungated_drain - mean_gated_drain, 8)
            if mean_gated_drain is not None and mean_ungated_drain is not None
            else None
        )

        return GovernorMeasurements(
            turns=len(records),
            gated_turns=len(gated),
            ungated_turns=len(ungated),
            mean_drain_per_turn=round(self._mean([r.delta_energy for r in records]) or 0.0, 8),
            mean_gated_drain=mean_gated_drain,
            mean_ungated_drain=mean_ungated_drain,
            mean_response_len=round(self._mean([r.response_len for r in records]) or 0.0, 2),
            mean_gated_response_len=self._mean([r.response_len for r in gated], digits=2),
            mean_ungated_response_len=self._mean([r.response_len for r in ungated], digits=2),
            estimated_gate_savings=estimated_savings,
            intervention_failures=intervention_failures,
            mode_distribution=modes,
            fitted_alpha=fitted_alpha,
        )

    def summary(self) -> Dict[str, object]:
        """Quick health check for trajectory analysis."""

        measurements = self.measurements().to_dict()
        if measurements["turns"] == 0:
            return {"status": "no data", "turns": 0}
        measurements["status"] = "ok"
        return measurements

    def _compute_mode(self, energy: float) -> str:
        if not self._valid_energy(energy):
            return "HOLD"

        if energy >= 1.0:
            return "NORMAL"

        for mode, (lo, hi) in ENERGY_THRESHOLDS.items():
            if lo <= energy < hi:
                return mode
        return "CRITICAL"

    def _verify_intervention(self, record: TurnRecord) -> None:
        """Closed-loop check: if gated, was drain below calibrated expectation?"""

        if not record.gate_applied or self._fitted_alpha is None:
            return

        expected_drain = self._fitted_alpha * record.response_len
        actual_drain = record.delta_energy
        if actual_drain > expected_drain * 1.20:
            self._intervention_failures += 1
            self._append_event(
                {
                    "event": "INTERVENTION_FAILED",
                    "turn": record.turn,
                    "gate_applied": True,
                    "expected_drain": round(expected_drain, 8),
                    "actual_drain": round(actual_drain, 8),
                    "response_len": record.response_len,
                    "note": "TOKEN_GATE did not reduce energy drain as predicted",
                }
            )

    def _append_calibration_log(self, record: TurnRecord) -> None:
        entry = {
            "event": "TURN",
            "timestamp": record.timestamp,
            "turn": record.turn,
            "prev_energy": round(record.prev_energy, 8),
            "new_energy": round(record.new_energy, 8),
            "delta_energy": round(record.delta_energy, 8),
            "response_len": record.response_len,
            "gate_applied": record.gate_applied,
            "energy_mode": record.energy_mode,
            "max_tokens": record.max_tokens,
            "interventions": record.interventions,
        }
        self._append_json(entry)

    def _append_event(self, event: Dict[str, object]) -> None:
        event = dict(event)
        event["timestamp"] = time.time()
        self._append_json(event)

    def _append_json(self, entry: Dict[str, object]) -> None:
        with self._calibration_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    @staticmethod
    def _valid_energy(energy: float) -> bool:
        try:
            value = float(energy)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and 0.0 <= value <= 1.0

    @staticmethod
    def _mean(values: List[float] | List[int], digits: int = 8) -> Optional[float]:
        if not values:
            return None
        return round(float(sum(values) / len(values)), digits)


def _self_check() -> bool:
    """Prove gate contracts, calibration, and verification paths."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir) / "governor_calibration.jsonl"
        gov = CognitiveGovernor(calibration_log=tmp)

        try:
            r1 = gov.apply(current_energy=0.80, turn=1)
            assert r1.energy_mode == "NORMAL"
            assert not r1.gate_applied
            assert r1.max_tokens == 1000
            print(f"  turn 1 [energy=0.80] mode={r1.energy_mode} max_tokens={r1.max_tokens}")

            r2 = gov.apply(current_energy=0.25, turn=2)
            assert r2.energy_mode == "LOW_POWER"
            assert r2.gate_applied
            assert r2.max_tokens == 400
            assert "TOKEN_GATE" in r2.interventions
            assert "MODE_ENTERED_LOW_POWER" in r2.interventions
            print(f"  turn 2 [energy=0.25] mode={r2.energy_mode} max_tokens={r2.max_tokens}")

            r3 = gov.apply(current_energy=0.10, turn=3)
            assert r3.energy_mode == "CRITICAL"
            assert r3.max_tokens == 150
            assert "ENERGY_MIN_REACHED" in r3.interventions
            print(f"  turn 3 [energy=0.10] mode={r3.energy_mode} max_tokens={r3.max_tokens}")

            r4 = gov.apply(current_energy=float("nan"), turn=4)
            assert r4.energy_mode == "HOLD"
            assert r4.max_tokens == 80
            assert "ENERGY_SENSOR_INVALID" in r4.interventions
            print(f"  turn 4 [energy=nan] mode={r4.energy_mode} max_tokens={r4.max_tokens}")

            gov._fitted_alpha = 0.00005
            gov.record_turn(prev_energy=0.80, new_energy=0.79, response_len=1000, gate_result=r1)
            gov.record_turn(prev_energy=0.25, new_energy=0.24, response_len=400, gate_result=r2)
            gov.record_turn(prev_energy=0.10, new_energy=0.099, response_len=150, gate_result=r3)
            print(f"  summary: {gov.summary()}")

            rng = np.random.default_rng(42)
            gov2 = CognitiveGovernor(calibration_log=Path(tmp_dir) / "synthetic.jsonl")
            for idx, response_len in enumerate(rng.integers(200, 1000, size=30), start=1):
                drain = 0.00002 * int(response_len) + float(rng.normal(0.0, 0.0004))
                gate = gov2.apply(current_energy=0.8, turn=idx)
                gov2.record_turn(
                    prev_energy=0.8,
                    new_energy=0.8 - drain,
                    response_len=int(response_len),
                    gate_result=gate,
                )
            alpha = gov2.calibrate_alpha(min_samples=20)
            assert alpha is not None
            assert 0.000015 <= alpha <= 0.000025
            print(f"  calibration returned alpha={alpha:.8f}")

            print("COGNITIVE GOVERNOR SELF-CHECK PASSED")
            return True
        except AssertionError as exc:
            print(f"SELF-CHECK FAILED: {exc}")
            return False


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_check() else 1)

