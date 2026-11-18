# PSC Scaled Engine

`psc_scaled.py` implements the vectorized PSC (Projection–Stability–Confidence)
engine used by DRIFT to forecast per-dimension crisis and fire alerts before a
dimension actually crosses a safety threshold. It replaces earlier boolean
"chaotic / not chaotic" plus fixed-horizon designs with three continuous
upgrades:

1. **Continuous chaos scoring** — every dimension gets a value in `[0, 1]`
   instead of a binary chaos flag.
2. **Per-dimension dynamic prediction horizons** — the number of steps used for
   forecasting shrinks as chaos grows.
3. **Residual-based confidence** — alerts only fire when the linear model fits
   recent history well, the trend is meaningful over the chosen horizon, and
   direction is consistent.

The engine is intentionally self-contained and CPU-only. It uses a fixed-size
circular buffer plus batched NumPy math so 16 DRIFT dimensions and 500 test
dimensions cost roughly the same per cycle.

## Dimensions and polarity

Every dimension is normalized to `[0, 1]`. The `DIMENSION_POLARITY` map
declares which side of the range is "safer":

- `+1` — higher is safer, crisis is near `0`. Examples: `focus`, `coherence`,
  `stability`, `clarity`, `energy`, `alignment`, `confidence`, `resilience`,
  `situational_awareness`, `task_progress`, `context_integrity`,
  `memory_coherence`.
- `-1` — lower is safer, crisis is near `1`. Examples: `threat_pressure`,
  `error_pressure`, `latency_pressure`, `resource_pressure`.

Extra dimensions can be added via the `polarity=` kwarg to
`PSCBatchEngine`. Unknown dimensions default to `+1`.

## Policies

Three built-in policies live in `POLICY_CONFIG`:

| Policy         | `crisis_threshold` | `confidence_threshold` | `min_bad_delta` | `recovery_margin` |
| -------------- | ------------------ | ---------------------- | --------------- | ----------------- |
| `SECURITY`     | 0.25               | 0.44                   | 0.015           | 0.10              |
| `BALANCED`     | 0.25               | 0.55                   | 0.025           | 0.12              |
| `CONSERVATIVE` | 0.25               | 0.66                   | 0.035           | 0.15              |

- `crisis_threshold` — distance from the safe edge (low side for `+1` polarity,
  high side for `-1`) that counts as "in crisis". The engine derives
  `low_side_threshold = crisis_threshold` and
  `high_side_threshold = 1 - crisis_threshold`.
- `confidence_threshold` — required residual-based confidence for an alert.
- `min_bad_delta` — minimum unsafe move between current and predicted values
  before alerting.
- `recovery_margin` — buffer above the crisis threshold before an active alert
  clears.

Override any of these per instance with `crisis_threshold=` and
`confidence_threshold=` on the constructor.

## Public surface

```python
from psc_scaled import PSCBatchEngine, DIMENSION_POLARITY

engine = PSCBatchEngine(
    dimensions=list(DIMENSION_POLARITY),
    policy="SECURITY",
    buffer_size=32,          # circular buffer capacity, must be >= min_history
    min_history=8,           # samples required before run() returns non-None
)

for reading in stream_of_states:  # each reading is {dim_name: value in [0,1]}
    engine.push_state(reading)
    result = engine.run()          # None while warming up
    if result is None:
        continue
    for i, dim in enumerate(result.dimensions):
        if result.alerted[i]:
            handle_alert(dim, result)
```

`PSCBatchResult` fields:

- `cycle` — 1-indexed cycle count.
- `dimensions` — tuple of dimension names in engine order.
- `current`, `predicted` — 1-D NumPy arrays of the latest and forecasted values.
- `alerted` — boolean array of alerts fired on **this** cycle only.
- `active_alerts` — boolean array of unresolved alerts across cycles.
- `chaos_scores` — continuous chaos scores per dimension.
- `n_steps_used` — the dynamic horizon actually used for each dimension.
- `confidence` — residual-based confidence per dimension.
- `crisis_boundaries` — the threshold each dimension is compared against
  (polarity-aware).
- `policy` — policy label carried through for logging.

## Chaos score (`_batch_chaos_score`)

For each dimension, the score combines four components over the last 16
samples:

- **Volatility** — `std(deltas) / 0.095`, clipped to `[0, 1]`.
- **Residual** — RMSE of a weighted linear fit divided by `0.115`.
- **Acceleration** — `std(diff(deltas)) / 0.14`.
- **Direction changes** — fraction of significant sign flips between
  consecutive deltas, where "significant" means the absolute delta clears
  `max(0.012, median(|delta|) * 0.45)`.

Weighted sum: `0.34*volatility + 0.28*residual + 0.18*acceleration + 0.20*direction_changes`.
The tests assert that smooth trending trajectories score much lower than noisy
trajectories with the same mean.

## Dynamic horizon (`_dynamic_n_steps`)

```
steps = round(max_steps - chaos^0.82 * (max_steps - min_steps))
```

with `min_steps=4`, `max_steps=10` by default. Stable dimensions look far
ahead; chaotic dimensions shrink to a short-horizon forecast.

## Projection (`_project_batch`)

A weighted linear fit is run over the last `min(16, buffer_len)` samples with
an EWLS-style age-based weight (`alpha = 2 / (window_len + 1) * (1 + 1.75 * chaos)`,
clipped to `[0.08, 0.62]`). The predicted value is
`current + slope * n_steps`, clipped to `[0, 1]`.

## Confidence (`_batch_residual_confidence`)

Blends three signals and applies a chaos penalty:

- **Residual score** — `exp(-rmse / 0.052)`. Sharp drop when the linear fit
  breaks down.
- **Trend score** — `clip(|slope| * n_steps / 0.075, 0, 1)`. Requires a
  meaningful span over the chosen horizon.
- **Direction score** — fraction of recent deltas whose sign matches the
  polarity-implied unsafe direction.

Combined: `0.50*residual + 0.34*trend + 0.16*direction`, then multiplied by
`(1 - 0.58 * chaos)`. Optional projected-distance boost when `predicted` is
provided.

## Alert logic

An alert fires when **all** of the following are true for a dimension:

1. The predicted value already crosses the crisis threshold in the unsafe
   direction (`projected_crisis`).
2. The slope is heading the unsafe way (`bad_slope`).
3. The polarity-adjusted delta from current to predicted is at least
   `min_bad_delta`.
4. Confidence is at least `confidence_threshold`.

Only newly-alerted dimensions appear in `alerted`; anything still in an
unresolved alert stays true in `active_alerts` until the dimension recovers by
`recovery_margin` past the crisis threshold. That prevents oscillation from
repeatedly firing during noisy crisis periods.

## Warm-up and buffer sizing

- `run()` returns `None` until the buffer has `min_history` samples. Push
  neutral defaults (`0.5`) before the loop if you need instant readiness.
- `buffer_size` bounds memory and how far back weighted fits can reach. The
  engine only looks at the last 16 samples anyway, so values much larger than
  16 mainly buy more history for external analytics.
- `_dimension_names(count)` gives you a stable, extendable name list if you
  want to benchmark synthetic dimension counts beyond the default map.

## Benchmarking scale

`benchmark_scale(dim_counts=(16, 50, 100, 200, 500), n_cycles=500, seed=42)`
runs the full push+run cycle on synthetic drift for each dimension count and
returns:

```
{
  16: {"mean_us": ..., "p99_us": ..., "cycles_per_sec": ..., "memory_bytes": ...},
  ...
}
```

The full end-to-end validation report (per-trajectory lead time, chaos
distribution at alert, horizon distribution, and scale table) lives in
`scripts/psc_scaled_validation.py`:

```bash
python3 scripts/psc_scaled_validation.py --no-figure
```

Add `--output path/to/psc_scaled_validation.png` (or omit `--no-figure`) to
render the four-panel matplotlib summary.

## Tuning tips

- If false positives are too high on noisy but bounded trajectories, raise
  `confidence_threshold` (or switch to `BALANCED`/`CONSERVATIVE`).
- If alerts fire late on slow linear drifts, lower `min_bad_delta` or reduce
  `crisis_threshold`.
- Non-standard dimensions with different value scales should still be
  normalized to `[0, 1]` before pushing. Do the scaling in the caller —
  `push_state` silently clips.
