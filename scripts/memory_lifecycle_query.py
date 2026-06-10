"""Report DRIFT memory lifecycle health from ChromaDB metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_timestamp(value: object) -> Optional[float]:
    """
    Convert a timestamp-like value into a Unix timestamp or return None if it cannot be interpreted.
    
    Parameters:
        value (object): A timestamp candidate. Accepted forms:
            - None or empty string: treated as missing.
            - int or float: interpreted as a numeric Unix timestamp (seconds).
            - string: either a numeric string (seconds) or an ISO 8601 datetime string (a trailing 'Z' is accepted).
    
    Returns:
        float | None: Unix timestamp in seconds if parsing succeeds, `None` otherwise.
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


def _retrieval_count(metadata: Mapping[str, object]) -> int:
    """
    Extract the numeric retrieval count from a metadata mapping by checking common keys.
    
    Parameters:
        metadata (Mapping[str, object]): Metadata mapping to inspect. The function checks keys in this order:
            "retrieval_count", "retrieved_count", "access_count", "times_retrieved".
    
    Returns:
        int: The parsed retrieval count. Returns 0 if none of the keys are present, the value is None,
             or conversion to int fails.
    """
    for key in ("retrieval_count", "retrieved_count", "access_count", "times_retrieved"):
        value = metadata.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _poor_session_only(metadata: Mapping[str, object]) -> bool:
    """
    Determines whether a memory's retrievals are exclusively associated with poor or failed sessions.
    
    Parameters:
        metadata (Mapping[str, object]): Metadata mapping for a memory. Recognized keys:
            - "poor_session_retrievals" or "failed_session_retrievals": count of poor/failed retrievals
            - "session_outcome": outcome label (e.g., "poor", "failed")
            - "session_success": boolean-like success indicator
            - retrieval count keys used by internal counting helpers (e.g., "retrieval_count", "access_count")
    
    Returns:
        bool: `true` if the metadata indicates the memory was retrieved only in poor/failed sessions, `false` otherwise.
    """
    poor = metadata.get("poor_session_retrievals", metadata.get("failed_session_retrievals", 0))
    total = _retrieval_count(metadata)
    outcome = str(metadata.get("session_outcome", "")).lower()
    success = metadata.get("session_success")

    if outcome in {"poor", "failed", "failure", "bad"}:
        return True
    if success is False or str(success).lower() == "false":
        return True
    try:
        return int(total) > 0 and int(poor) >= int(total)
    except (TypeError, ValueError):
        return False


def build_lifecycle_report(
    metadatas: Iterable[Mapping[str, object]],
    *,
    now: Optional[datetime] = None,
    stale_days: int = 30,
    frequent_threshold: int = 5,
) -> dict[str, object]:
    """
    Generate a lifecycle health report for a collection of memory metadata.
    
    Parameters:
        metadatas (Iterable[Mapping[str, object]]): Iterable of metadata mappings for memories.
        now (Optional[datetime]): Reference time used to evaluate staleness; defaults to current time.
        stale_days (int): Age threshold in days for marking a memory as stale.
        frequent_threshold (int): Retrieval-count threshold for classifying a memory as frequently retrieved.
    
    Returns:
        dict[str, object]: A report containing counts, ratios, thresholds, and ID lists:
            - total_memories (int): Number of metadata items processed.
            - stale_days (int): The `stale_days` value used.
            - stale_count (int): Number of memories older than the cutoff.
            - stale_ratio (float): Fraction of memories that are stale (rounded to 4 decimals).
            - missing_last_accessed_count (int): Number of memories without a parseable `last_accessed`.
            - frequent_threshold (int): The `frequent_threshold` value used.
            - frequently_retrieved_count (int): Number of memories with retrieval count >= threshold.
            - frequent_only_poor_session_count (int): Count of frequently retrieved memories that only appear in poor/failed sessions.
            - stale_ids (list): IDs of memories considered stale.
            - missing_last_accessed_ids (list): IDs of memories missing parseable `last_accessed`.
            - frequent_only_poor_session_ids (list): IDs of frequently retrieved memories associated only with poor/failed sessions.
    """

    metadata_list = [dict(metadata or {}) for metadata in metadatas]
    now = now or datetime.now()
    cutoff = (now - timedelta(days=stale_days)).timestamp()

    missing_last_accessed = []
    stale = []
    frequently_retrieved = []
    frequent_only_poor = []

    for index, metadata in enumerate(metadata_list):
        memory_id = metadata.get("id", metadata.get("memory_id", index))
        last_accessed = _as_timestamp(metadata.get("last_accessed"))
        retrieved = _retrieval_count(metadata)

        if last_accessed is None:
            missing_last_accessed.append(memory_id)
        elif last_accessed < cutoff:
            stale.append(memory_id)

        if retrieved >= frequent_threshold:
            frequently_retrieved.append(memory_id)
            if _poor_session_only(metadata):
                frequent_only_poor.append(memory_id)

    total = len(metadata_list)
    return {
        "total_memories": total,
        "stale_days": stale_days,
        "stale_count": len(stale),
        "stale_ratio": round(len(stale) / total, 4) if total else 0.0,
        "missing_last_accessed_count": len(missing_last_accessed),
        "frequent_threshold": frequent_threshold,
        "frequently_retrieved_count": len(frequently_retrieved),
        "frequent_only_poor_session_count": len(frequent_only_poor),
        "stale_ids": stale,
        "missing_last_accessed_ids": missing_last_accessed,
        "frequent_only_poor_session_ids": frequent_only_poor,
    }


def load_chroma_metadatas(path: str, collection_name: str) -> list[Mapping[str, object]]:
    """
    Load all metadata documents from a persistent ChromaDB collection.
    
    Parameters:
        path (str): Filesystem path used to open the ChromaDB persistent client.
        collection_name (str): Name of the collection to read metadata from.
    
    Returns:
        list[Mapping[str, object]]: A list of metadata mappings from the collection; an empty list if no metadatas are present.
    
    Raises:
        SystemExit: If the `chromadb` package is not installed.
    """

    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit(
            "chromadb is not installed. Install it or use --metadata-json for an offline report."
        ) from exc

    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    return list(results.get("metadatas") or [])


def load_json_metadatas(path: Path) -> list[Mapping[str, object]]:
    """
    Load ChromaDB/DRIFT metadata entries from a JSON file.
    
    Parameters:
        path (Path): Path to a JSON file containing either a top-level list of metadata objects
            or an object with a "metadatas" key mapping to such a list.
    
    Returns:
        list[Mapping[str, object]]: The list of metadata mappings parsed from the file.
    
    Raises:
        ValueError: If the JSON is neither a list nor an object with a "metadatas" list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("metadatas", [])
    if not isinstance(data, list):
        raise ValueError("metadata JSON must be a list or an object with a 'metadatas' list")
    return data


def main() -> None:
    """
    Run the CLI: load ChromaDB or JSON metadata, build a lifecycle health report, and print human-readable summaries plus the full JSON report.
    
    Parses command-line options for ChromaDB path/collection, stale-days, frequent-threshold, and an optional offline metadata JSON file. Loads metadata from the chosen source, generates a lifecycle report using those thresholds, prints three summary lines (stale counts, missing last_accessed counts, frequently retrieved-only-in-poor-sessions counts), and outputs the full report as formatted JSON.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/chroma", help="ChromaDB persistent path")
    parser.add_argument("--collection", default="drift_memory", help="Chroma collection name")
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument("--frequent-threshold", type=int, default=5)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional JSON fixture with Chroma-style metadatas for offline analysis",
    )
    args = parser.parse_args()

    if args.metadata_json:
        metadatas = load_json_metadatas(args.metadata_json)
    else:
        metadatas = load_chroma_metadatas(args.path, args.collection)

    report = build_lifecycle_report(
        metadatas,
        stale_days=args.stale_days,
        frequent_threshold=args.frequent_threshold,
    )

    print(
        "Memories not accessed in "
        f"{report['stale_days']} days: {report['stale_count']}/{report['total_memories']}"
    )
    print(
        "Memories missing last_accessed metadata: "
        f"{report['missing_last_accessed_count']}/{report['total_memories']}"
    )
    print(
        "Frequently retrieved memories only seen in poor sessions: "
        f"{report['frequent_only_poor_session_count']}/{report['frequently_retrieved_count']}"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
