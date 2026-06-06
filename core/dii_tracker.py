"""Minimal DII tracker with an explicit startup heartbeat."""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List


_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass(frozen=True)
class DIIReading:
    """One Dynamic Integration Index sensor reading."""

    value: float
    interaction_len: int
    timestamp: float = field(default_factory=time.time)


class DIITracker:
    """Small variance-bearing DII sensor.

    Before the first heartbeat, ``get_current()`` returns the neutral safety
    value of ``0.5``. Calling ``update_from_interaction`` wakes the sensor and
    records a deterministic reading from the interaction text.
    """

    def __init__(self, default_value: float = 0.5) -> None:
        self.default_value = float(default_value)
        self._lock = threading.Lock()
        self._readings: List[DIIReading] = []

    def update_from_interaction(self, text: str) -> float:
        reading = DIIReading(
            value=self._score_interaction(text),
            interaction_len=len(text or ""),
        )
        with self._lock:
            self._readings.append(reading)
        return reading.value

    def get_current(self) -> float:
        with self._lock:
            if not self._readings:
                return self.default_value
            return self._readings[-1].value

    def is_awake(self) -> bool:
        with self._lock:
            return bool(self._readings)

    def variance(self) -> float:
        with self._lock:
            values = [reading.value for reading in self._readings]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def summary(self) -> dict[str, float | int | bool]:
        with self._lock:
            samples = len(self._readings)
            current = self._readings[-1].value if self._readings else self.default_value
        return {
            "awake": samples > 0,
            "samples": samples,
            "current": round(current, 6),
            "variance": round(self.variance(), 8),
        }

    def reset(self) -> None:
        with self._lock:
            self._readings.clear()

    @staticmethod
    def _score_interaction(text: str) -> float:
        words = _WORD_RE.findall(text or "")
        if not words:
            return 0.5

        unique_ratio = len({word.lower() for word in words}) / len(words)
        length_signal = min(len(text) / 240.0, 1.0)
        structure_signal = min(
            (text.count(".") + text.count("?") + text.count("!") + text.count("\n")) / 8.0,
            1.0,
        )
        score = 0.30 + 0.35 * unique_ratio + 0.20 * length_signal + 0.15 * structure_signal
        if not math.isfinite(score):
            return 0.5
        return max(0.0, min(1.0, score))


_TRACKER = DIITracker()


def get_dii_tracker() -> DIITracker:
    return _TRACKER


__all__ = ["DIIReading", "DIITracker", "get_dii_tracker"]
