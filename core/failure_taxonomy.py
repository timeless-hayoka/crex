"""Failure taxonomy helpers for DRIFT trajectory logs."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Iterable, Mapping, Optional


class FailureType(str, Enum):
    """Structured failure labels for turn-level trajectory metadata."""

    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    MATH_HALLUCINATION = "math_hallucination"
    SEAM_NOT_FOUND = "seam_not_found"
    DIVISION_BY_ZERO = "division_by_zero"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_INVALID = "sensor_invalid"
    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_STALE = "retrieval_stale"
    UPDATE_READ_PATH_MISMATCH = "update_read_path_mismatch"
    LATENCY_SPIKE = "latency_spike"
    UNKNOWN = "unknown"


_PATTERNS: tuple[tuple[FailureType, tuple[str, ...]], ...] = (
    (FailureType.DIVISION_BY_ZERO, ("division by zero", "zerodivisionerror")),
    (
        FailureType.QUOTA_EXCEEDED,
        ("quota exceeded", "resource exhausted", "rate limit", "429"),
    ),
    (
        FailureType.MATH_HALLUCINATION,
        ("math hallucination", "arithmetic hallucination", "incorrect arithmetic"),
    ),
    (
        FailureType.SEAM_NOT_FOUND,
        (
            "coordination plugin instance not found",
            "object has no attribute",
            "attributeerror",
            "module not found",
            "importerror",
            "seam not found",
        ),
    ),
    (
        FailureType.UPDATE_READ_PATH_MISMATCH,
        ("ablation stub", "update layer", "read path", "wrong path"),
    ),
    (
        FailureType.SENSOR_STUCK,
        ("locked at", "zero variance", "constant sensor", "sensor stuck"),
    ),
    (
        FailureType.SENSOR_INVALID,
        ("nan", "sensor invalid", "invalid sensor", "not finite"),
    ),
    (
        FailureType.RETRIEVAL_EMPTY,
        ("retrieval count 0", "no memories retrieved", "empty retrieval"),
    ),
    (
        FailureType.RETRIEVAL_STALE,
        ("stale memory", "calcifying", "not accessed"),
    ),
    (FailureType.TIMEOUT, ("timeout", "timed out", "deadline exceeded")),
    (FailureType.LATENCY_SPIKE, ("latency spike", "slow turn")),
)


def normalize_failure_type(value: object) -> Optional[str]:
    """Return a canonical failure type value, or ``None`` for empty input."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    for failure_type in FailureType:
        if text == failure_type.value:
            return failure_type.value
    return FailureType.UNKNOWN.value


def classify_failure(
    raw_text: str = "",
    *,
    latency_seconds: Optional[float] = None,
    timeout_seconds: float = 10.0,
) -> Optional[str]:
    """Classify a raw failure message or turn metrics into a taxonomy value."""

    if latency_seconds is not None and latency_seconds >= timeout_seconds:
        return FailureType.TIMEOUT.value

    text = raw_text.lower()
    if not text.strip():
        return None

    for failure_type, needles in _PATTERNS:
        if any(needle in text for needle in needles):
            return failure_type.value

    if any(marker in text for marker in ("error", "exception", "failed", "traceback")):
        return FailureType.UNKNOWN.value
    return None


def with_failure_type(
    metadata: Mapping[str, object],
    *,
    raw_text: str = "",
    latency_seconds: Optional[float] = None,
    timeout_seconds: float = 10.0,
) -> dict:
    """Return turn metadata with a ``failure_type`` field added."""

    enriched = dict(metadata)
    explicit = normalize_failure_type(enriched.get("failure_type"))
    inferred = classify_failure(
        raw_text,
        latency_seconds=latency_seconds,
        timeout_seconds=timeout_seconds,
    )
    enriched["failure_type"] = explicit or inferred
    return enriched


def summarize_failure_types(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Count non-empty failure types in trajectory-like records."""

    counts: Counter[str] = Counter()
    for record in records:
        failure_type = normalize_failure_type(record.get("failure_type"))
        if failure_type:
            counts[failure_type] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "FailureType",
    "classify_failure",
    "normalize_failure_type",
    "summarize_failure_types",
    "with_failure_type",
]
