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
    for key in ("retrieval_count", "retrieved_count", "access_count", "times_retrieved"):
        value = metadata.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _poor_session_only(metadata: Mapping[str, object]) -> bool:
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
    """Build stale, missing-access, and poor-session retrieval counts."""

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
    """Load all metadata from a ChromaDB collection."""

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
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("metadatas", [])
    if not isinstance(data, list):
        raise ValueError("metadata JSON must be a list or an object with a 'metadatas' list")
    return data


def main() -> None:
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
