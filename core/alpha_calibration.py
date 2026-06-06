"""Alpha calibration for response-length energy drain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class AlphaCalibrationResult:
    samples: int
    alpha: Optional[float]
    correlation: Optional[float]
    r_squared: Optional[float]
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_alpha(records: Iterable[Mapping[str, object]], min_samples: int = 20) -> AlphaCalibrationResult:
    """Fit ``delta_energy = alpha * response_len`` with no intercept."""

    clean_records = []
    for record in records:
        try:
            response_len = float(record["response_len"])
            if "delta_energy" in record:
                delta_energy = float(record["delta_energy"])
            else:
                delta_energy = float(record["prev_energy"]) - float(record["new_energy"])
        except (KeyError, TypeError, ValueError):
            continue
        if response_len > 0 and math.isfinite(response_len) and math.isfinite(delta_energy):
            clean_records.append((response_len, delta_energy))

    if len(clean_records) < min_samples:
        return AlphaCalibrationResult(
            samples=len(clean_records),
            alpha=None,
            correlation=None,
            r_squared=None,
            verdict=f"insufficient_data:{len(clean_records)}/{min_samples}",
        )

    lengths = np.asarray([record[0] for record in clean_records], dtype=np.float64)
    drains = np.asarray([record[1] for record in clean_records], dtype=np.float64)
    if lengths.std() < 1.0:
        return AlphaCalibrationResult(
            samples=len(clean_records),
            alpha=None,
            correlation=None,
            r_squared=None,
            verdict="no_response_length_variance",
        )

    alpha = float(np.dot(lengths, drains) / np.dot(lengths, lengths))
    predicted = alpha * lengths
    residual = drains - predicted
    ss_res = float(np.dot(residual, residual))
    centered = drains - drains.mean()
    ss_tot = float(np.dot(centered, centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    correlation = float(np.corrcoef(lengths, drains)[0, 1])
    if not math.isfinite(correlation):
        correlation = None

    verdict = "line_detected" if correlation is not None and abs(correlation) >= 0.70 else "shotgun_blast"
    return AlphaCalibrationResult(
        samples=len(clean_records),
        alpha=alpha,
        correlation=correlation,
        r_squared=r_squared,
        verdict=verdict,
    )


__all__ = ["AlphaCalibrationResult", "fit_alpha"]
