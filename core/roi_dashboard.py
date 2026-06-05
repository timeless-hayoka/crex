"""Subsystem ROI aggregation for DRIFT trajectory records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SubsystemROI:
    subsystem: str
    samples: int
    mean_cost_ms: float
    mean_benefit: float | None
    roi_per_ms: float | None
    benefit_label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _benefit_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def _iter_subsystem_entries(record: Mapping[str, object]):
    subsystems = record.get("subsystems")
    if isinstance(subsystems, Mapping):
        for name, value in subsystems.items():
            if isinstance(value, Mapping):
                yield str(name), value
            else:
                yield str(name), {"benefit": value}

    costs = record.get("subsystem_costs_ms")
    if isinstance(costs, Mapping):
        benefits = record.get("subsystem_benefits", {})
        for name, cost in costs.items():
            benefit = benefits.get(name) if isinstance(benefits, Mapping) else None
            yield str(name), {"cost_ms": cost, "benefit": benefit}

    name = record.get("subsystem")
    if name:
        yield str(name), record


def build_roi_dashboard(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate cost and benefit metrics per subsystem."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"costs": [], "benefits": []})

    for record in records:
        for subsystem, entry in _iter_subsystem_entries(record):
            cost = _as_float(entry.get("cost_ms", entry.get("latency_ms")))
            benefit = _as_float(entry.get("benefit", entry.get("benefit_score")))
            if cost is not None:
                grouped[subsystem]["costs"].append(cost)
            if benefit is not None:
                grouped[subsystem]["benefits"].append(benefit)
            if cost is None and benefit is None:
                grouped[subsystem]["costs"]

    rows = []
    for subsystem, values in grouped.items():
        costs = values["costs"]
        benefits = values["benefits"]
        samples = max(len(costs), len(benefits), 1)
        mean_cost = sum(costs) / len(costs) if costs else 0.0
        mean_benefit = sum(benefits) / len(benefits) if benefits else None
        roi = mean_benefit / mean_cost if mean_benefit is not None and mean_cost > 0 else None
        rows.append(
            SubsystemROI(
                subsystem=subsystem,
                samples=samples,
                mean_cost_ms=round(mean_cost, 4),
                mean_benefit=round(mean_benefit, 4) if mean_benefit is not None else None,
                roi_per_ms=round(roi, 6) if roi is not None else None,
                benefit_label=_benefit_label(mean_benefit),
            ).to_dict()
        )

    return sorted(rows, key=lambda row: (row["benefit_label"] == "unknown", row["subsystem"]))


def load_jsonl_records(path: Path) -> list[Mapping[str, object]]:
    """Load records from JSON array, JSON object, or JSONL trajectory logs."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("records", data.get("turns", [data]))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


__all__ = ["SubsystemROI", "build_roi_dashboard", "load_jsonl_records"]
