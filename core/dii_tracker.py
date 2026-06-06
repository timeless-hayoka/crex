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
        """
        Initialize the tracker with a neutral default value and prepare internal thread-safe storage.
        
        Parameters:
            default_value (float): Neutral DII value returned by get_current() when no readings exist; will be converted to float.
        """
        self.default_value = float(default_value)
        self._lock = threading.Lock()
        self._readings: List[DIIReading] = []

    def update_from_interaction(self, text: str) -> float:
        """
        Record a new DII reading computed from the provided interaction text and return its score.
        
        Appends the new immutable reading to the tracker's history.
        
        Parameters:
            text (str): Interaction text to score. `None` or an empty string is treated as no words.
        
        Returns:
            float: The computed DII score for the provided text (clamped to the range 0.0–1.0).
        """
        reading = DIIReading(
            value=self._score_interaction(text),
            interaction_len=len(text or ""),
        )
        with self._lock:
            self._readings.append(reading)
        return reading.value

    def get_current(self) -> float:
        """
        Retrieve the most recent DII value, or the tracker's default value when no readings exist.
        
        Returns:
            float: The last recorded reading's `value`, or the tracker's `default_value` if there are no readings.
        """
        with self._lock:
            if not self._readings:
                return self.default_value
            return self._readings[-1].value

    def is_awake(self) -> bool:
        """
        Indicates whether the tracker has recorded at least one reading.
        
        Returns:
            `true` if at least one reading exists, `false` otherwise.
        """
        with self._lock:
            return bool(self._readings)

    def variance(self) -> float:
        """
        Compute the population variance of the tracked DII reading values.
        
        Returns:
            float: Population variance of the stored reading `value`s; returns 0.0 if fewer than 2 samples are available.
        """
        with self._lock:
            values = [reading.value for reading in self._readings]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def summary(self) -> dict[str, float | int | bool]:
        """
        Produce a snapshot of the tracker's state including awake status, sample count, current value, and variance.
        
        Returns:
            dict[str, float | int | bool]: A mapping with the following keys:
                - "awake" (bool): True if at least one reading is stored, False otherwise.
                - "samples" (int): Number of stored readings.
                - "current" (float): Most recent reading value (or the tracker's default if no readings), rounded to 6 decimals.
                - "variance" (float): Population variance of all stored reading values, rounded to 8 decimals.
        """
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
        """
        Resets the tracker to its initial state by clearing all stored readings.
        
        This operation is performed under the tracker's lock and is safe to call concurrently.
        """
        with self._lock:
            self._readings.clear()

    @staticmethod
    def _score_interaction(text: str) -> float:
        """
        Estimate a Dynamic Integration Index (DII) score for an interaction text based on lexical diversity, length, and structural punctuation.
        
        Higher values indicate greater lexical diversity, longer text, and more structural sentence punctuation/newlines; lower values indicate the opposite. Empty or invalid text yields a neutral value.
        
        Parameters:
            text (str): The interaction text to score.
        
        Returns:
            float: A score between 0.0 and 1.0. `0.5` is used as a neutral fallback for empty or invalid input.
        """
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
    """
    Retrieve the module-level Dynamic Integration Index tracker instance.
    
    Returns:
    	The singleton DIITracker instance.
    """
    return _TRACKER


__all__ = ["DIIReading", "DIITracker", "get_dii_tracker"]
