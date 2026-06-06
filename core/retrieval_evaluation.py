"""Retrieval evaluation metrics that only require memory metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional


def as_timestamp(value: object) -> Optional[float]:
    """
    Convert a metadata value to a Unix timestamp when possible.
    
    Accepts ints/floats, numeric strings, or ISO-8601 timestamp strings (a trailing "Z" is accepted).
    Returns None for None, empty strings, or values that cannot be parsed.
    
    Parameters:
        value (object): Metadata value to parse as a timestamp.
    
    Returns:
        Optional[float]: Unix timestamp in seconds as a float if parsing succeeds, `None` otherwise.
    """

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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def retrieval_count(metadata: Mapping[str, object]) -> int:
    """
    Selects and returns the most appropriate retrieval/access count from metadata.
    
    Parameters:
        metadata (Mapping[str, object]): Mapping of metadata fields. The function checks the keys
            "retrieval_count", "retrieved_count", "access_count", and "times_retrieved" in that order.
    
    Returns:
        int: Non-negative retrieval count from the first present key. Returns 0 if no count keys are present
        or if the found value cannot be converted to an integer.
    """

    for key in ("retrieval_count", "retrieved_count", "access_count", "times_retrieved"):
        value = metadata.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
    return 0


def memory_id(metadata: Mapping[str, object], fallback: int) -> object:
    """
    Selects an identifier for a memory from metadata, falling back to the provided value.
    
    Parameters:
        metadata (Mapping[str, object]): Metadata mapping to read from.
        fallback (int): Value to return if neither `id` nor `memory_id` is present in metadata.
    
    Returns:
        The first present value among `metadata['id']`, `metadata['memory_id']`, or `fallback`.
    """
    return metadata.get("id", metadata.get("memory_id", fallback))


def build_retrieval_report(
    metadatas: Iterable[Mapping[str, object]],
    *,
    now: Optional[datetime] = None,
    repeat_threshold: int = 5,
) -> dict[str, object]:
    """
    Compute per-memory retrieval statistics and an age distribution summary for a collection of metadata records.
    
    Parameters:
        metadatas (Iterable[Mapping[str, object]]): Iterable of metadata mappings; each item may contain retrieval and creation keys.
        now (Optional[datetime]): Reference time used to compute ages; naive datetimes are treated as UTC. If omitted, the current time is used.
        repeat_threshold (int): Minimum retrieval count for a memory to be considered "repeated".
    
    Returns:
        dict[str, object]: A report containing:
            - "total_memories" (int): Number of metadata items processed.
            - "retrieved_memory_count" (int): Count of memories with retrievals > 0.
            - "retrieved_memory_ratio" (float): Retrieved count divided by total memories, rounded to 4 decimals (0.0 if total is 0).
            - "total_retrievals" (int): Sum of all retrieval counts.
            - "repeat_threshold" (int): The threshold passed through.
            - "repeated_memory_count" (int): Number of memories with retrieval_count >= repeat_threshold.
            - "repeat_retrieval_ratio" (float): Fraction of retrievals attributable to repeated memories (rounded to 4 decimals; 0.0 if total_retrievals is 0).
            - "memory_repeat_frequency" (list[dict]): Sorted list of repeated memories with entries {"memory_id": id, "retrieval_count": count}, sorted by descending count then ascending memory_id string.
            - "memory_age_distribution" (dict): Buckets for creation age with keys "0_7_days", "8_30_days", "31_90_days", "over_90_days", and "unknown".
    """

    metadata_list = [dict(metadata or {}) for metadata in metadatas]
    now_dt = now or datetime.now()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()

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
