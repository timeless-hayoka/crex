# Cognitive Governor and Alpha Calibration

`core/cognitive_governor.py` and `core/alpha_calibration.py` together implement
DRIFT's behavior gate: they translate a scalar energy telemetry signal into a
concrete generation-time constraint (max tokens per response), and they measure
whether that constraint actually reduced energy drain in production.

## Data model

- **`CognitiveGovernor`** — stateful, thread-safe controller. Constructor
  arguments:
  - `calibration_log: str | Path = "logs/governor_calibration.jsonl"` — JSONL
    append-only audit trail; parent directory is created eagerly.
  - `energy_min: float = 0.10` — energy value at or below which the gate emits
    an `ENERGY_MIN_REACHED` intervention regardless of mode.
- **`GateResult`** — one decision returned by `apply()`: `max_tokens`,
  `energy_mode`, `mode_entered`, `interventions`, `gate_applied`,
  `energy_snapshot`, `turn`.
- **`TurnRecord`** — one recorded turn: previous/new energy, delta,
  `response_len`, gate applied?, energy mode, interventions list, timestamp.
- **`GovernorMeasurements`** — aggregated closed-loop measurements returned by
  `measurements()`.

## Energy modes

The stepped mode table (from `ENERGY_MODES` / `ENERGY_THRESHOLDS`):

| Energy range      | Mode        | `max_tokens` | Notes                                    |
| ----------------- | ----------- | ------------ | ---------------------------------------- |
| `> 0.50`          | `NORMAL`    | 1000         | Full cognitive capacity                  |
| `0.30 < e ≤ 0.50` | `MODERATE`  | 700          | Keep responses bounded                   |
| `0.15 < e ≤ 0.30` | `LOW_POWER` | 400          | Conserving; longer responses deferred    |
| `0.00 ≤ e ≤ 0.15` | `CRITICAL`  | 150          | Minimal output; recovering               |
| non-finite / OOB  | `HOLD`      | 80           | Sensor invalid; respond but do not act   |

The `HOLD` mode is triggered whenever `energy` is `NaN`, infinite, or outside
`[0, 1]`. `_calculate_safe_token_limit(current_energy)` mirrors this table for
external callers that need the same ladder without touching the governor.

## Per-turn contract

```python
from core.cognitive_governor import CognitiveGovernor

gov = CognitiveGovernor(calibration_log="logs/governor.jsonl")

# Before inference:
gate = gov.apply(current_energy=0.42, turn=turn_index)
response = model.generate(prompt, max_tokens=gate.max_tokens)

# After homeostasis updates the energy:
gov.record_turn(
    prev_energy=0.42,
    new_energy=0.38,
    response_len=len(response),
    gate_result=gate,
)
```

`apply()` returns a `GateResult` **before** the response is generated so that
`gate.max_tokens` can be passed straight to the generation API and
`gate.interventions` can be logged alongside the turn.

Interventions emitted by `apply()`:

- `ENERGY_SENSOR_INVALID` — `apply()` received non-finite or out-of-range energy.
- `ENERGY_MIN_REACHED` — energy is at or below `energy_min`.
- `MODE_ENTERED_<MODE>` — the current mode changed on this turn.
- `TOKEN_GATE` — the returned `max_tokens` is lower than the `NORMAL` budget
  (i.e. the gate is actually constraining generation).

## Alpha calibration

The governor logs enough per-turn data to fit the linear no-intercept model

```
delta_energy = alpha * response_len
```

Two entry points are available:

1. **In-process**: `governor.calibrate_alpha(min_samples=20)` uses only
   **ungated** turns (`gate_applied=False`) so the fit is not biased by the
   very intervention we are trying to measure. It:
   - Requires at least `min_samples` ungated turns.
   - Requires `response_len` variance ≥ 1.0 (diverse prompts).
   - Prints a human-readable summary and stores `alpha` on the governor.
   - Appends an `ALPHA_CALIBRATED` event to the calibration log.
2. **Offline**: `core.alpha_calibration.fit_alpha(records, min_samples=20)`
   accepts any iterable of dict-like turn records with `response_len` and
   either `delta_energy` or `prev_energy`/`new_energy`. It returns an
   `AlphaCalibrationResult`:

   | Field | Meaning |
   | --- | --- |
   | `samples` | number of clean records that passed validation |
   | `alpha` | fitted `alpha` (`None` when the fit was rejected) |
   | `correlation` | Pearson correlation between `response_len` and drain |
   | `r_squared` | `1 - SS_res / SS_tot` for the no-intercept fit |
   | `verdict` | `line_detected` when `abs(correlation) >= 0.70`, else `shotgun_blast`, or `insufficient_data:N/M` / `no_response_length_variance` when the fit was rejected |

   The offline path is what `scripts/alpha_calibration_run.py` uses.

## Closed-loop verification

Once `_fitted_alpha` is set (either via `calibrate_alpha()` or assigned
directly), every subsequent `record_turn(...)` call runs `_verify_intervention`.
For gated turns:

```
expected_drain = fitted_alpha * response_len
if actual_drain > expected_drain * 1.20:
    _intervention_failures += 1
    log INTERVENTION_FAILED
```

That is: after calibration, if a gated turn drains more than 20% above the
linear expectation, it counts as a failure. `GovernorMeasurements.intervention_failures`
is the running count; each failure is also written to the calibration log as
a JSONL `INTERVENTION_FAILED` event with the expected/actual drain.

## Measurements

`governor.measurements()` returns a `GovernorMeasurements`:

- `turns`, `gated_turns`, `ungated_turns`
- `mean_drain_per_turn`, `mean_gated_drain`, `mean_ungated_drain`
- `mean_response_len`, `mean_gated_response_len`, `mean_ungated_response_len`
- `estimated_gate_savings = mean_ungated_drain - mean_gated_drain`
- `intervention_failures`
- `mode_distribution` — histogram of `energy_mode` values seen
- `fitted_alpha`

`governor.summary()` calls `measurements()` and adds a `status` field
(`"no data"` or `"ok"`).

Empty gated / ungated buckets return `None` (rather than 0) so downstream
consumers can distinguish "no data" from "measured 0".

## Calibration log format

Each line in the calibration log is a self-contained JSON object. Two event
kinds are used:

`TURN` — appended on every `record_turn()`:

```json
{
  "event": "TURN",
  "timestamp": 1770000000.0,
  "turn": 42,
  "prev_energy": 0.42, "new_energy": 0.4185,
  "delta_energy": 0.0015,
  "response_len": 300,
  "gate_applied": true,
  "energy_mode": "LOW_POWER",
  "max_tokens": 400,
  "interventions": ["MODE_ENTERED_LOW_POWER", "TOKEN_GATE"]
}
```

`ALPHA_CALIBRATED` — appended on a successful `calibrate_alpha()`:

```json
{
  "event": "ALPHA_CALIBRATED",
  "samples": 30,
  "alpha": 2.009e-05,
  "correlation": 0.998,
  "timestamp": 1770000123.4
}
```

`INTERVENTION_FAILED` — appended on a gate verification miss:

```json
{
  "event": "INTERVENTION_FAILED",
  "turn": 57,
  "gate_applied": true,
  "expected_drain": 0.0075,
  "actual_drain": 0.0120,
  "response_len": 300,
  "note": "TOKEN_GATE did not reduce energy drain as predicted",
  "timestamp": 1770000200.5
}
```

## Operational scripts

- `scripts/alpha_calibration_run.py` — end-to-end calibration run. Either
  reads a real trajectory (`--trajectory-jsonl PATH`) or generates a
  50-turn synthetic gated trajectory with a known `alpha`, then prints a
  verdict from `fit_alpha`. Requires at least 20 clean records.
- `scripts/cognitive_governor_measurements.py` — synthetic 80-turn simulation
  that runs a 30-turn ungated calibration block followed by a stress block
  with binding token gates. Prints mode distribution, drain means, estimated
  and counterfactual gate savings, intervention failures, and the fitted α
  error against the injected `true_alpha`.

Both scripts fail with a non-zero exit if their preconditions (min samples,
turn counts) are not met.

## Testing hooks

- `python3 core/cognitive_governor.py` runs `_self_check()`, exercising the
  gate ladder, `HOLD` on `NaN`, calibration on a synthetic 30-turn stream,
  and asserts the fitted α lands in `[1.5e-5, 2.5e-5]` for the deterministic
  seed.
- `tests/test_cognitive_governor.py`, `tests/test_alpha_calibration.py`
  cover the gate ladder, calibration verdicts, and the closed-loop
  measurement bookkeeping.
