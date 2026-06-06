"""Startup hooks for DRIFT telemetry sensors."""

from __future__ import annotations

from core.dii_tracker import get_dii_tracker


def initialize_sensors(
    baseline: str = "System initialization sequence complete.",
) -> dict[str, object]:
    """
    Initialize DRIFT telemetry sensors and update the DII tracker with a startup baseline.
    
    Parameters:
        baseline (str): Message used to update the DII tracker state during initialization.
    
    Returns:
        dict[str, object]: A dictionary with keys:
            - 'dii_awake': `True` if the DII tracker reports it is awake, `False` otherwise.
            - 'dii_current': The value returned by the tracker's `update_from_interaction` call.
            - 'dii_samples': The current sample count from the tracker's summary (value of `summary()['samples']`).
    """

    print("[*] Initializing telemetry sensors...")
    dii = get_dii_tracker()
    value = dii.update_from_interaction(baseline)
    print("[+] DII Sensor online and tracking.")
    return {
        "dii_awake": dii.is_awake(),
        "dii_current": value,
        "dii_samples": dii.summary()["samples"],
    }


__all__ = ["initialize_sensors"]
