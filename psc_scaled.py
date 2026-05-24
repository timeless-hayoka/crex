"""Vectorized PSC scaled engine.

This module implements the V4 PSC upgrades used by the validation simulation:

* continuous chaos scoring instead of a binary chaos flag
* per-dimension dynamic prediction horizons
* residual-based confidence for alert gating

The implementation is intentionally self-contained and CPU-only.  It uses a
small circular history buffer and vectorized NumPy math so the 16 DRIFT
dimensions can be evaluated in one batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


# Polarity describes which side of a normalized [0, 1] dimension is safer.
#   1: higher values are safer; crisis is projected near 0.
#  -1: lower values are safer; crisis is projected near 1.
DIMENSION_POLARITY: Dict[str, int] = {
    "focus": 1,
    "coherence": 1,
    "stability": 1,
    "clarity": 1,
    "energy": 1,
    "alignment": 1,
    "confidence": 1,
    "resilience": 1,
    "situational_awareness": 1,
    "task_progress": 1,
    "context_integrity": 1,
    "memory_coherence": 1,
    "threat_pressure": -1,
    "error_pressure": -1,
    "latency_pressure": -1,
    "resource_pressure": -1,
}


POLICY_CONFIG: Mapping[str, Mapping[str, float]] = {
    # SECURITY catches credible degradation early while confidence still has to
    # clear the residual/chaos gate.
    "SECURITY": {
        "crisis_threshold": 0.25,
        "confidence_threshold": 0.44,
        "min_bad_delta": 0.015,
        "recovery_margin": 0.10,
    },
    "BALANCED": {
        "crisis_threshold": 0.25,
        "confidence_threshold": 0.55,
        "min_bad_delta": 0.025,
        "recovery_margin": 0.12,
    },
    "CONSERVATIVE": {
        "crisis_threshold": 0.25,
        "confidence_threshold": 0.66,
        "min_bad_delta": 0.035,
        "recovery_margin": 0.15,
    },
}


@dataclass(frozen=True)
class PSCBatchResult:
    """Result returned by :class:`PSCBatchEngine`.

    ``alerted`` contains newly fired alerts for this cycle. ``active_alerts``
    remains true for dimensions that are still inside an unrecovered alert
    state, which keeps noisy crisis periods from repeatedly firing.
    """

    cycle: int
    dimensions: Sequence[str]
    current: np.ndarray
    predicted: np.ndarray
    alerted: np.ndarray
    active_alerts: np.ndarray
    chaos_scores: np.ndarray
    n_steps_used: np.ndarray
    confidence: np.ndarray
    crisis_boundaries: np.ndarray
    policy: str


class PSCStateBuffer:
    """Fixed-size circular buffer for normalized PSC state vectors."""

    def __init__(
        self,
        dimensions: Sequence[str],
        capacity: int = 32,
        default_value: float = 0.5,
    ) -> None:
        if capacity < 8:
            raise ValueError("PSCStateBuffer capacity must be at least 8 samples")
        if not dimensions:
            raise ValueError("PSCStateBuffer requires at least one dimension")

        self.dimensions: List[str] = list(dimensions)
        self.capacity = int(capacity)
        self.default_value = float(default_value)
        self._index = {name: i for i, name in enumerate(self.dimensions)}
        self._data = np.full(
            (self.capacity, len(self.dimensions)),
            self.default_value,
            dtype=np.float64,
        )
        self._write_index = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def data(self) -> np.ndarray:
        return self._data

    def push(self, state: Mapping[str, float]) -> np.ndarray:
        """Push one state sample and return the normalized vector stored."""

        if self._count:
            row = self.values(limit=1)[-1].copy()
        else:
            row = np.full(len(self.dimensions), self.default_value, dtype=np.float64)

        for name, value in state.items():
            if name not in self._index:
                continue
            row[self._index[name]] = np.clip(float(value), 0.0, 1.0)

        self._data[self._write_index] = row
        self._write_index = (self._write_index + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        return row

    def values(self, limit: Optional[int] = None) -> np.ndarray:
        """Return buffered samples in chronological order."""

        count = self._count if limit is None else min(int(limit), self._count)
        if count <= 0:
            return np.empty((0, len(self.dimensions)), dtype=np.float64)

        start = (self._write_index - count) % self.capacity
        if start + count <= self.capacity:
            return self._data[start : start + count].copy()

        first = self._data[start:]
        second = self._data[: count - len(first)]
        return np.vstack((first, second))

    def last(self) -> np.ndarray:
        if self._count == 0:
            return np.full(len(self.dimensions), self.default_value, dtype=np.float64)
        return self.values(limit=1)[-1]


def _as_history_matrix(history: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(history, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("history must be a 2-D array shaped (time, dimensions)")
    if arr.shape[0] == 0:
        raise ValueError("history must contain at least one sample")
    return np.clip(arr, 0.0, 1.0)


def _fit_linear(history: np.ndarray, weights: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return vectorized slope and intercept for each dimension."""

    y = _as_history_matrix(history)
    n, dims = y.shape
    x = np.arange(n, dtype=np.float64)[:, None]

    if weights is None:
        w = np.ones((n, dims), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = np.repeat(w[:, None], dims, axis=1)
        if w.shape != y.shape:
            raise ValueError("weights must match history shape")

    w_sum = np.maximum(w.sum(axis=0), 1e-12)
    x_mean = (w * x).sum(axis=0) / w_sum
    y_mean = (w * y).sum(axis=0) / w_sum
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.maximum((w * x_centered * x_centered).sum(axis=0), 1e-12)
    slope = (w * x_centered * y_centered).sum(axis=0) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _linear_residual_rmse(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = _as_history_matrix(history)
    slope, intercept = _fit_linear(y)
    x = np.arange(y.shape[0], dtype=np.float64)[:, None]
    fitted = intercept + slope * x
    rmse = np.sqrt(np.mean((y - fitted) ** 2, axis=0))
    return slope, rmse


def _batch_chaos_score(history: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return a continuous chaos score in [0, 1] for each dimension.

    The score combines volatility, linear-model residuals, acceleration, and
    direction changes.  It is deliberately continuous: stable trends, feedback
    oscillations, and high-noise states land at different points on the scale
    instead of collapsing into a binary "chaotic" flag.
    """

    y = _as_history_matrix(history)
    if y.shape[0] < 4:
        return np.zeros(y.shape[1], dtype=np.float64)

    recent = y[-min(16, y.shape[0]) :]
    deltas = np.diff(recent, axis=0)
    abs_delta = np.abs(deltas)

    volatility = np.clip(np.std(deltas, axis=0) / 0.095, 0.0, 1.0)

    _, rmse = _linear_residual_rmse(recent)
    residual = np.clip(rmse / 0.115, 0.0, 1.0)

    if deltas.shape[0] > 1:
        accel = np.diff(deltas, axis=0)
        acceleration = np.clip(np.std(accel, axis=0) / 0.14, 0.0, 1.0)

        # Ignore tiny derivative flips so measurement jitter does not dominate
        # otherwise smooth trajectories.
        median_step = np.median(abs_delta, axis=0)
        sign_floor = np.maximum(0.012, median_step * 0.45)
        significant = abs_delta >= sign_floor
        signs = np.sign(np.where(significant, deltas, 0.0))
        changes = (signs[1:] * signs[:-1]) < 0.0
        valid_pairs = significant[1:] & significant[:-1]
        direction_changes = changes.sum(axis=0) / np.maximum(valid_pairs.sum(axis=0), 1)
    else:
        acceleration = np.zeros(recent.shape[1], dtype=np.float64)
        direction_changes = np.zeros(recent.shape[1], dtype=np.float64)

    chaos = (
        0.34 * volatility
        + 0.28 * residual
        + 0.18 * acceleration
        + 0.20 * direction_changes
    )
    return np.clip(chaos, 0.0, 1.0)


def _dynamic_n_steps(
    chaos_scores: np.ndarray | Sequence[float],
    *,
    min_steps: int = 4,
    max_steps: int = 10,
) -> np.ndarray:
    """Map continuous chaos scores to per-dimension forecast horizons."""

    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError("expected 0 < min_steps <= max_steps")

    chaos = np.clip(np.asarray(chaos_scores, dtype=np.float64), 0.0, 1.0)
    curved = np.power(chaos, 0.82)
    steps = np.rint(max_steps - curved * (max_steps - min_steps)).astype(np.int64)
    return np.clip(steps, min_steps, max_steps)


def _adaptive_alpha(window_len: int, chaos_scores: np.ndarray) -> np.ndarray:
    """Return per-dimension EWLS alpha using the actual projection window."""

    if window_len < 2:
        raise ValueError("window_len must be at least 2")

    chaos = np.clip(np.asarray(chaos_scores, dtype=np.float64), 0.0, 1.0)
    base = 2.0 / (window_len + 1.0)
    alpha = base * (1.0 + 1.75 * chaos)
    return np.clip(alpha, 0.08, 0.62)


def _projection_weights(window_len: int, chaos_scores: np.ndarray) -> np.ndarray:
    alpha = _adaptive_alpha(window_len, chaos_scores)
    age = np.arange(window_len - 1, -1, -1, dtype=np.float64)[:, None]
    return np.power(1.0 - alpha[None, :], age)


def _project_batch(
    history: np.ndarray | Sequence[Sequence[float]],
    n_steps: np.ndarray | Sequence[int],
    chaos_scores: Optional[np.ndarray | Sequence[float]] = None,
    *,
    min_history: int = 8,
    max_window: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each dimension forward by its own horizon.

    Window enforcement happens here, where the projection math actually uses
    the history length. This avoids dead "effective length" calculations in the
    alpha function that do not change the data being fitted.
    """

    y = _as_history_matrix(history)
    if y.shape[0] < min_history:
        raise ValueError(f"projection requires at least {min_history} samples")

    window_len = min(max_window, y.shape[0])
    window_len = max(min_history, window_len)
    recent = y[-window_len:]

    if chaos_scores is None:
        chaos = _batch_chaos_score(recent)
    else:
        chaos = np.clip(np.asarray(chaos_scores, dtype=np.float64), 0.0, 1.0)

    steps = np.asarray(n_steps, dtype=np.float64)
    if steps.ndim == 0:
        steps = np.full(recent.shape[1], float(steps), dtype=np.float64)
    if steps.shape[0] != recent.shape[1]:
        raise ValueError("n_steps length must match the number of dimensions")

    weights = _projection_weights(window_len, chaos)
    slope, _ = _fit_linear(recent, weights)
    predicted = np.clip(recent[-1] + slope * steps, 0.0, 1.0)
    return predicted, slope


def _batch_residual_confidence(
    history: np.ndarray | Sequence[Sequence[float]],
    predicted: Optional[np.ndarray | Sequence[float]] = None,
    n_steps: Optional[np.ndarray | Sequence[int]] = None,
    polarity: Optional[np.ndarray | Sequence[int]] = None,
    chaos_scores: Optional[np.ndarray | Sequence[float]] = None,
) -> np.ndarray:
    """Compute residual-based forecast confidence for each dimension.

    Confidence rises when the recent trajectory has low linear residuals, a
    meaningful trend over the selected horizon, and consistent direction.  It
    falls continuously as chaos increases.
    """

    y = _as_history_matrix(history)
    recent = y[-min(16, y.shape[0]) :]
    dims = recent.shape[1]

    slope, rmse = _linear_residual_rmse(recent)
    residual_score = np.exp(-rmse / 0.052)

    if n_steps is None:
        if chaos_scores is None:
            chaos_scores = _batch_chaos_score(recent)
        steps = _dynamic_n_steps(chaos_scores)
    else:
        steps = np.asarray(n_steps, dtype=np.float64)
        if steps.ndim == 0:
            steps = np.full(dims, float(steps), dtype=np.float64)

    trend_span = np.abs(slope) * steps
    trend_score = np.clip(trend_span / 0.075, 0.0, 1.0)

    deltas = np.diff(recent, axis=0)
    if deltas.shape[0] == 0:
        direction_score = np.zeros(dims, dtype=np.float64)
    else:
        if polarity is None:
            expected_direction = np.sign(slope)
        else:
            pol = np.asarray(polarity, dtype=np.float64)
            expected_direction = -np.sign(pol)
        meaningful = np.abs(deltas) >= np.maximum(0.010, np.std(deltas, axis=0) * 0.18)
        aligned = np.sign(deltas) == expected_direction[None, :]
        direction_score = (aligned & meaningful).sum(axis=0) / np.maximum(
            meaningful.sum(axis=0), 1
        )

    if chaos_scores is None:
        chaos = _batch_chaos_score(recent)
    else:
        chaos = np.clip(np.asarray(chaos_scores, dtype=np.float64), 0.0, 1.0)

    base = 0.50 * residual_score + 0.34 * trend_score + 0.16 * direction_score
    chaos_penalty = 1.0 - 0.58 * chaos
    confidence = base * chaos_penalty

    if predicted is not None:
        pred = np.asarray(predicted, dtype=np.float64)
        if pred.shape[0] != dims:
            raise ValueError("predicted length must match the number of dimensions")
        projection_distance = np.abs(pred - recent[-1])
        confidence *= np.clip(0.55 + projection_distance / 0.18, 0.55, 1.0)

    return np.clip(confidence, 0.0, 1.0)


class PSCBatchEngine:
    """Batch PSC engine with circular buffers and vectorized forecasts."""

    def __init__(
        self,
        dimensions: Sequence[str],
        *,
        policy: str = "SECURITY",
        buffer_size: int = 32,
        min_history: int = 8,
        polarity: Optional[Mapping[str, int]] = None,
        crisis_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        if policy not in POLICY_CONFIG:
            raise ValueError(f"unknown PSC policy {policy!r}")
        if min_history < 4:
            raise ValueError("min_history must be at least 4")
        if buffer_size < min_history:
            raise ValueError("buffer_size must be >= min_history")

        self.dimensions = list(dimensions)
        self.policy = policy
        self.buffer = PSCStateBuffer(self.dimensions, capacity=buffer_size)
        self.min_history = int(min_history)
        self._cycle = 0

        merged_polarity: MutableMapping[str, int] = dict(DIMENSION_POLARITY)
        if polarity:
            merged_polarity.update(polarity)
        self.polarity = np.array(
            [1 if merged_polarity.get(dim, 1) >= 0 else -1 for dim in self.dimensions],
            dtype=np.int8,
        )

        config = POLICY_CONFIG[policy]
        threshold = (
            float(crisis_threshold)
            if crisis_threshold is not None
            else float(config["crisis_threshold"])
        )
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(config["confidence_threshold"])
        )
        self.min_bad_delta = float(config["min_bad_delta"])
        self.recovery_margin = float(config["recovery_margin"])
        self.low_side_threshold = threshold
        self.high_side_threshold = 1.0 - threshold
        self.crisis_boundaries = np.where(
            self.polarity > 0,
            self.low_side_threshold,
            self.high_side_threshold,
        )
        self._active_alerts = np.zeros(len(self.dimensions), dtype=bool)

    def push_state(self, state: Mapping[str, float]) -> np.ndarray:
        self._cycle += 1
        return self.buffer.push(state)

    def run(self) -> Optional[PSCBatchResult]:
        if self.buffer.count < self.min_history:
            return None

        history = self.buffer.values()
        current = history[-1]
        chaos_scores = _batch_chaos_score(history)
        n_steps = _dynamic_n_steps(chaos_scores)
        predicted, slope = _project_batch(
            history,
            n_steps,
            chaos_scores,
            min_history=self.min_history,
        )
        confidence = _batch_residual_confidence(
            history,
            predicted=predicted,
            n_steps=n_steps,
            polarity=self.polarity,
            chaos_scores=chaos_scores,
        )

        projected_crisis = np.where(
            self.polarity > 0,
            predicted <= self.low_side_threshold,
            predicted >= self.high_side_threshold,
        )
        current_crisis = np.where(
            self.polarity > 0,
            current <= self.low_side_threshold,
            current >= self.high_side_threshold,
        )
        bad_delta = np.where(self.polarity > 0, current - predicted, predicted - current)
        bad_slope = np.where(self.polarity > 0, slope < 0.0, slope > 0.0)

        raw_alerts = (
            projected_crisis
            & bad_slope
            & (bad_delta >= self.min_bad_delta)
            & (confidence >= self.confidence_threshold)
        )
        new_alerts = raw_alerts & ~self._active_alerts

        recovered = np.where(
            self.polarity > 0,
            current >= self.low_side_threshold + self.recovery_margin,
            current <= self.high_side_threshold - self.recovery_margin,
        )
        self._active_alerts = (self._active_alerts | new_alerts | current_crisis) & (
            ~recovered | raw_alerts | current_crisis
        )

        return PSCBatchResult(
            cycle=self._cycle,
            dimensions=tuple(self.dimensions),
            current=current.copy(),
            predicted=predicted.copy(),
            alerted=new_alerts.copy(),
            active_alerts=self._active_alerts.copy(),
            chaos_scores=chaos_scores.copy(),
            n_steps_used=n_steps.astype(np.int64, copy=True),
            confidence=confidence.copy(),
            crisis_boundaries=self.crisis_boundaries.copy(),
            policy=self.policy,
        )


def _dimension_names(count: int) -> List[str]:
    base = list(DIMENSION_POLARITY.keys())
    if count <= len(base):
        return base[:count]
    return base + [f"dim_{idx}" for idx in range(len(base), count)]


def benchmark_scale(
    dim_counts: Iterable[int] = (16, 50, 100, 200, 500),
    *,
    n_cycles: int = 500,
    seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    """Benchmark end-to-end push+run cost for different dimension counts."""

    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive")

    rng = np.random.default_rng(seed)
    results: Dict[int, Dict[str, float]] = {}

    for dim_count in dim_counts:
        dim_count = int(dim_count)
        if dim_count <= 0:
            raise ValueError("dimension counts must be positive")

        dims = _dimension_names(dim_count)
        engine = PSCBatchEngine(dims, policy="SECURITY")
        state = rng.uniform(0.48, 0.86, size=dim_count)
        drift = rng.normal(-0.0005, 0.0015, size=dim_count)
        timings_ns: List[int] = []
        total_cycles = n_cycles + engine.min_history + 8

        for cycle in range(total_cycles):
            state = np.clip(
                state + drift + rng.normal(0.0, 0.012, size=dim_count),
                0.0,
                1.0,
            )
            sample = {name: float(value) for name, value in zip(dims, state)}

            start = perf_counter_ns()
            engine.push_state(sample)
            engine.run()
            elapsed = perf_counter_ns() - start

            if cycle >= engine.min_history + 8:
                timings_ns.append(elapsed)

        timings = np.asarray(timings_ns, dtype=np.float64) / 1_000.0
        mean_us = float(np.mean(timings))
        p99_us = float(np.percentile(timings, 99))
        results[dim_count] = {
            "mean_us": mean_us,
            "p99_us": p99_us,
            "cycles_per_sec": float(1_000_000.0 / mean_us) if mean_us > 0 else float("inf"),
            "memory_bytes": float(engine.buffer.data.nbytes),
        }

    return results


__all__ = [
    "DIMENSION_POLARITY",
    "PSCBatchEngine",
    "PSCBatchResult",
    "PSCStateBuffer",
    "_batch_chaos_score",
    "_batch_residual_confidence",
    "_dynamic_n_steps",
    "benchmark_scale",
]
