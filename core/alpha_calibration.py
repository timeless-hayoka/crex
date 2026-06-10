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
        """
        Convert the dataclass to a dictionary mapping field names to their values.
        
        Returns:
            dict[str, object]: A dictionary where keys are the dataclass field names and values are the corresponding field values.
        """
        return asdict(self)


def fit_alpha(records: Iterable[Mapping[str, object]], min_samples: int = 20) -> AlphaCalibrationResult:
    """
    Estimate the slope (alpha) of a no-intercept linear relationship between response length and observed energy drain.
    
    Parses an iterable of record mappings to extract numeric `response_len` and `delta_energy` (or derives `delta_energy` from `prev_energy - new_energy`), filters out invalid or non-finite entries, and fits the model `delta_energy = alpha * response_len` using least squares with no intercept. The function also computes Pearson correlation and an R² based on residuals and returns a verdict indicating fit quality.
    
    Parameters:
        records (Iterable[Mapping[str, object]]): Iterable of mapping-like records. Each record must contain a numeric `"response_len"` and either `"delta_energy"` or both `"prev_energy"` and `"new_energy"`.
        min_samples (int): Minimum number of valid records required to attempt fitting. If the cleaned sample count is less than this, no fit is performed.
    
    Returns:
        AlphaCalibrationResult: Dataclass with fields:
            - samples: number of cleaned records used,
            - alpha: fitted slope or `None` if fitting was not performed,
            - correlation: Pearson correlation between `response_len` and `delta_energy` or `None` if not finite,
            - r_squared: residual-based R² for the no-intercept fit or `None` when not computable,
            - verdict: one of:
                - `"insufficient_data:<actual>/<min_samples>"` when sample count is below `min_samples`,
                - `"no_response_length_variance"` when response lengths lack required variance,
                - `"line_detected"` when a finite correlation with absolute value >= 0.70 was found,
                - `"shotgun_blast"` otherwise.
    """

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
