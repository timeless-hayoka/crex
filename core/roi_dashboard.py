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
        """
        Return a plain dictionary representation of this SubsystemROI suitable for serialization.
        
        Returns:
            A dictionary mapping dataclass field names (`subsystem`, `samples`, `mean_cost_ms`, `mean_benefit`, `roi_per_ms`, `benefit_label`) to their values.
        """
        return asdict(self)


def _as_float(value: object) -> float | None:
    """
    Convert the input to a numeric value when possible, treating absent or empty inputs as missing.
    
    Parameters:
        value: The input to convert; may be None, an empty string, a number, or a string representation of a number.
    
    Returns:
        The converted float if conversion succeeds; `None` if `value` is `None`, the empty string, or cannot be converted to a number.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _benefit_label(score: float | None) -> str:
    """
    Map a benefit score to a categorical label indicating its magnitude.
    
    Parameters:
        score (float | None): Benefit score to classify; `None` indicates an unknown score.
    
    Returns:
        str: One of `"unknown"`, `"high"`, `"medium"`, `"low"`, or `"none"`:
            - `"unknown"` if `score` is `None`
            - `"high"` if `score >= 0.70`
            - `"medium"` if `score >= 0.35`
            - `"low"` if `score > 0.0`
            - `"none"` otherwise
    """
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
    """
    Yield subsystem entries from a record supporting multiple accepted schema shapes.
    
    The function recognizes three shapes and yields (subsystem_name, entry) pairs:
    1. If `record["subsystems"]` is a mapping, yields each `name` with its associated mapping value; if a value is not a mapping, yields `{"benefit": value}` as the entry.
    2. If `record["subsystem_costs_ms"]` is a mapping, yields each `name` with an entry `{"cost_ms": cost, "benefit": benefit}` where `benefit` is taken from `record["subsystem_benefits"]` when available and is a mapping.
    3. Otherwise, if `record["subsystem"]` is present and truthy, yields `(str(record["subsystem"]), record)`.
    
    Parameters:
        record (Mapping[str, object]): A single trajectory record that may contain subsystem data in any of the supported shapes.
    
    Yields:
        tuple[str, Mapping[str, object]]: Pairs of subsystem name (string) and an entry mapping describing cost/benefit or the original record.
    """
    subsystems = record.get("subsystems")
    if isinstance(subsystems, Mapping):
        for name, value in subsystems.items():
            if isinstance(value, Mapping):
                yield str(name), value
            else:
                yield str(name), {"benefit": value}
        return

    costs = record.get("subsystem_costs_ms")
    if isinstance(costs, Mapping):
        benefits = record.get("subsystem_benefits", {})
        for name, cost in costs.items():
            benefit = benefits.get(name) if isinstance(benefits, Mapping) else None
            yield str(name), {"cost_ms": cost, "benefit": benefit}
        return

    name = record.get("subsystem")
    if name:
        yield str(name), record


def build_roi_dashboard(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """
    Aggregate cost and benefit metrics per subsystem from DRIFT trajectory records.
    
    Parameters:
        records (Iterable[Mapping[str, object]]): Iterable of record mappings. Each record may contain subsystem information in one of several supported shapes (e.g., a "subsystems" mapping, parallel "subsystem_costs_ms"/"subsystem_benefits" mappings, or a top-level "subsystem" entry).
    
    Returns:
        list[dict[str, object]]: A list of dictionaries, one per subsystem, sorted with rows labeled `"unknown"` first then by subsystem name. Each dictionary contains:
            - "subsystem" (str): Subsystem identifier.
            - "samples" (int): Number of samples used for aggregation (at least 1).
            - "mean_cost_ms" (float): Mean cost/latency in milliseconds (rounded to 4 decimals).
            - "mean_benefit" (float | None): Mean benefit score if any samples were present (rounded to 4 decimals) or `None`.
            - "roi_per_ms" (float | None): Ratio of mean benefit to mean cost when computable (rounded to 6 decimals) or `None`.
            - "benefit_label" (str): Categorical label for the mean benefit ("unknown", "high", "medium", "low", or "none").
    """

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

    return sorted(rows, key=lambda row: (not (row["benefit_label"] == "unknown"), row["subsystem"]))


def load_jsonl_records(path: Path) -> list[Mapping[str, object]]:
    """
    Load trajectory records from a file containing JSON, a JSON array/object, or newline-delimited JSON (JSONL).
    
    Reads the file as UTF-8 and returns an empty list for empty files. If the file begins with `[` or `{`, attempts to parse the whole text as JSON: if it yields a list that list is returned; if it yields a dict returns `dict["records"]` or `dict["turns"]` when present, otherwise a single-element list containing the dict. If JSON parsing is not applicable, the file is treated as line-delimited JSON and each non-empty line is parsed as a separate record.
    
    Parameters:
        path (Path): Path to the input file.
    
    Returns:
        list[Mapping[str, object]]: A list of record mappings parsed from the file.
    """

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("records", data.get("turns", [data]))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


__all__ = ["SubsystemROI", "build_roi_dashboard", "load_jsonl_records"]
