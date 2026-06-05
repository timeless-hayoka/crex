"""Startup hooks for DRIFT telemetry sensors."""

from __future__ import annotations

from core.dii_tracker import get_dii_tracker


def initialize_sensors(
    baseline: str = "System initialization sequence complete.",
) -> dict[str, object]:
    """Wake telemetry sensors before the consciousness loop begins."""

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
