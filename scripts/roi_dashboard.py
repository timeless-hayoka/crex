"""Render a subsystem ROI dashboard from trajectory-style JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.roi_dashboard import build_roi_dashboard, load_jsonl_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, help="JSON, JSONL, or trajectory records file")
    args = parser.parse_args()

    rows = build_roi_dashboard(load_jsonl_records(args.records))

    print(f"{'Subsystem':<16} {'Cost ms':>10} {'Benefit':>10} {'ROI/ms':>10} {'Label':>10}")
    print("-" * 62)
    for row in rows:
        benefit = "unknown" if row["mean_benefit"] is None else f"{row['mean_benefit']:.4f}"
        roi = "unknown" if row["roi_per_ms"] is None else f"{row['roi_per_ms']:.6f}"
        print(
            f"{row['subsystem']:<16} {row['mean_cost_ms']:>10.4f} "
            f"{benefit:>10} {roi:>10} {row['benefit_label']:>10}"
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
