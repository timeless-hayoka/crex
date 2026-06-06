"""Retrieval evaluation metrics that only require memory metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional


def as_timestamp(value: object) -> Optional[float]:
    """Parse numeric or ISO timestamp metadata values."""

    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return _to_utc(datetime.fromisoformat(text.replace("Z", "+00:00"))).timestamp()
    except ValueError:
        return None


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def retrieval_count(metadata: Mapping[str, object]) -> int:
    """Return the best available retrieval/access count from memory metadata."""

    for key in ("retrieval_count", "retrieved_count", "access_count", "times_retrieved"):
        value = metadata.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
    return 0


def memory_id(metadata: Mapping[str, object], fallback: int) -> object:
    return metadata.get("id", metadata.get("memory_id", fallback))


def build_retrieval_report(
    metadatas: Iterable[Mapping[str, object]],
    *,
    now: Optional[datetime] = None,
    repeat_threshold: int = 5,
) -> dict[str, object]:
    """Compute repeat-frequency and age-distribution metrics for memories."""

    metadata_list = [dict(metadata or {}) for metadata in metadatas]
    now_ts = _to_utc(now or datetime.now(timezone.utc)).timestamp()

    total_retrievals = 0
    retrieved = 0
    repeated = []
    age_buckets = {
        "0_7_days": 0,
        "8_30_days": 0,
        "31_90_days": 0,
        "over_90_days": 0,
        "unknown": 0,
    }

    for index, metadata in enumerate(metadata_list):
        count = retrieval_count(metadata)
        total_retrievals += count
        if count > 0:
            retrieved += 1
        if count >= repeat_threshold:
            repeated.append(
                {
                    "memory_id": memory_id(metadata, index),
                    "retrieval_count": count,
                }
            )

        created = as_timestamp(
            metadata.get("created_at", metadata.get("created", metadata.get("timestamp")))
        )
        if created is None:
            age_buckets["unknown"] += 1
            continue

        age_days = max(0.0, (now_ts - created) / 86_400.0)
        if age_days <= 7:
            age_buckets["0_7_days"] += 1
        elif age_days <= 30:
            age_buckets["8_30_days"] += 1
        elif age_days <= 90:
            age_buckets["31_90_days"] += 1
        else:
            age_buckets["over_90_days"] += 1

    repeated.sort(key=lambda item: (-int(item["retrieval_count"]), str(item["memory_id"])))
    total = len(metadata_list)
    repeat_retrievals = sum(max(0, item["retrieval_count"] - 1) for item in repeated)

    return {
        "total_memories": total,
        "retrieved_memory_count": retrieved,
        "retrieved_memory_ratio": round(retrieved / total, 4) if total else 0.0,
        "total_retrievals": total_retrievals,
        "repeat_threshold": repeat_threshold,
        "repeated_memory_count": len(repeated),
        "repeat_retrieval_ratio": (
            round(repeat_retrievals / total_retrievals, 4) if total_retrievals else 0.0
        ),
        "memory_repeat_frequency": repeated,
        "memory_age_distribution": age_buckets,
    }


__all__ = [
    "as_timestamp",
    "build_retrieval_report",
    "memory_id",
    "retrieval_count",
]
