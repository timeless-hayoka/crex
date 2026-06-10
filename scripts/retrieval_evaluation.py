"""Report retrieval repeat and memory age metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieval_evaluation import build_retrieval_report  # noqa: E402
from scripts.memory_lifecycle_query import load_chroma_metadatas, load_json_metadatas  # noqa: E402


def main() -> None:
    """
    CLI entry point that builds and prints a retrieval report containing repetition and memory-age metrics.
    
    Parses command-line options:
    - --path: ChromaDB persistent path (default "data/chroma").
    - --collection: Chroma collection name (default "drift_memory").
    - --repeat-threshold: Integer threshold used to classify repeated memories (default 5).
    - --metadata-json: Optional Path to a JSON fixture of Chroma-style metadatas for offline analysis.
    
    Loads metadatas from the provided JSON fixture when given, otherwise from ChromaDB using the provided path and collection, then builds a retrieval report and prints:
    - a summary line with repeated memory count and threshold,
    - the retrieved memory ratio,
    - the full report as pretty-printed JSON to standard output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/chroma", help="ChromaDB persistent path")
    parser.add_argument("--collection", default="drift_memory", help="Chroma collection name")
    parser.add_argument("--repeat-threshold", type=int, default=5)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional JSON fixture with Chroma-style metadatas for offline analysis",
    )
    args = parser.parse_args()
    if args.repeat_threshold < 1:
        parser.error("repeat-threshold must be >= 1")

    if args.repeat_threshold < 1:
        parser.error("repeat-threshold must be >= 1")

    if args.metadata_json:
        metadatas = load_json_metadatas(args.metadata_json)
    else:
        metadatas = load_chroma_metadatas(args.path, args.collection)

    report = build_retrieval_report(
        metadatas,
        repeat_threshold=args.repeat_threshold,
    )
    print(
        "Repeated memories: "
        f"{report['repeated_memory_count']}/{report['total_memories']} "
        f"(threshold={report['repeat_threshold']})"
    )
    print(f"Retrieved memory ratio: {report['retrieved_memory_ratio']}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
