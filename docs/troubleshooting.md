# Troubleshooting and Common Pitfalls

Short list of the setup and runtime issues you're most likely to hit when
working with CREX / DRIFT modules.

## Setup

### `ModuleNotFoundError: No module named 'numpy'`
NumPy is the only hard runtime dependency. Install it into your virtualenv:

```bash
python3 -m pip install numpy
```

Everything else (`pytest`, `matplotlib`, `chromadb`) is optional.

### `ModuleNotFoundError: No module named 'pytest'`
Only needed to run the test suite. Install with `pip install pytest`, then run
`python3 -m pytest tests/ -q`.

### `ModuleNotFoundError: No module named 'matplotlib'`
Only `scripts/psc_scaled_validation.py` uses matplotlib, and only when it
renders the summary figure. Either install matplotlib or pass `--no-figure`:

```bash
python3 scripts/psc_scaled_validation.py --no-figure
```

### `chromadb is not installed. Install it or use --metadata-json ...`
Emitted by `scripts/memory_lifecycle_query.py` and
`scripts/retrieval_evaluation.py` when they try to open a live ChromaDB path
without the `chromadb` package. Options:

- Install with `pip install chromadb`.
- Run offline with `--metadata-json PATH` and any JSON fixture matching the
  Chroma `metadatas` shape (see `tests/fixtures/memory_metadatas.json`).

### `import psc_scaled` fails when running from a nested directory
Every entrypoint script inserts the repo root into `sys.path`. If you're
running a raw `python3 -c "import psc_scaled"` from elsewhere, either `cd` to
the repo root or set `PYTHONPATH=/path/to/crex`.

## PSC engine

### `PSCBatchEngine.run()` returns `None`
By design: `run()` returns `None` until the buffer holds `min_history`
samples (default 8). Warm up the engine by pushing initial states — pushing
copies of a neutral vector such as `{dim: 0.5}` is fine.

### `ValueError: history must contain at least one sample`
Something upstream cleared the buffer. Confirm you're using a single
`PSCBatchEngine` per stream and not recreating it every cycle.

### Alerts never fire on a slow monotonic drift
Two knobs to look at:

- `min_bad_delta` — the smallest polarity-adjusted move from current to
  predicted that will alert. Increase it to be less sensitive; decrease it
  for slow drift. Overridable via a custom policy or by constructing the
  engine with `crisis_threshold=`/`confidence_threshold=` overrides.
- `confidence_threshold` — increase for fewer, higher-quality alerts;
  decrease when the linear fit isn't quite reaching the current threshold.

See the "Tuning tips" section of [`psc_engine.md`](psc_engine.md) for the
common failure modes.

### False alerts on noisy but stable trajectories
Switch to `BALANCED` or `CONSERVATIVE` policy, which raise both
`confidence_threshold` and `min_bad_delta`. The
`test_engine_suppresses_stable_noisy_false_alerts` test is a good reference
for what "suppresses" looks like in practice.

### `ValueError: PSCStateBuffer capacity must be at least 8 samples`
`buffer_size` must be ≥ `min_history` and ≥ 8. Keep buffer_size at 32 (the
default) unless you know why you need something different — the engine only
uses the last 16 samples anyway.

### Extra / non-standard dimensions
`DIMENSION_POLARITY` covers 16 canonical DRIFT dimensions. If you add your
own, pass a `polarity` mapping to the constructor so the engine knows which
side is unsafe. Unknown dimensions default to `+1` (higher is safer).

## Cognitive governor

### Governor stuck in `HOLD`
`HOLD` fires when `apply()` receives a value that isn't a finite number in
`[0, 1]`. Common causes:

- Sensor threading race — a `NaN` slipped in before initialization. Call
  `initialize_sensors()` once at startup.
- Integer overflow in an upstream normalization. Clip the value with
  `max(0.0, min(1.0, value))` before passing to `apply()`.

### `calibrate_alpha()` returns `None`
Three preconditions must hold:

1. At least `min_samples` **ungated** turns (default 20). Gated turns are
   excluded on purpose so the fit isn't biased by the intervention.
2. `response_len` variance ≥ 1.0 across those ungated turns — the fit needs
   diverse response lengths.
3. All lengths must be > 0 with a positive `sum(lengths**2)`.

If any check fails, `calibrate_alpha` prints the reason and returns `None`.
The offline `fit_alpha` in `core.alpha_calibration` returns the same signal
via `AlphaCalibrationResult.verdict` (`insufficient_data:N/M`,
`no_response_length_variance`, `shotgun_blast`, or `line_detected`).

### `verdict: shotgun_blast`
The fit produced an `alpha` but `abs(correlation) < 0.70`. That means
`response_len` isn't a strong predictor of drain, which usually means:

- Response length is capped by external constraints (e.g. all turns hit the
  same `max_tokens`).
- Energy is being modified by something other than response length between
  turns (e.g. background decay).
- Sample size is too small to see through the noise.

Widen the input variety and re-calibrate. Do **not** use a shotgun-blast
`alpha` for closed-loop verification — the ratio check `actual > 1.2 *
expected` becomes noise.

### `INTERVENTION_FAILED` events appearing after calibration
By design. It means a gated turn drained more than 20% above what
`alpha * response_len` predicts. Usually one of:

- The response was shorter than expected but a lot of energy still drained
  (external cost — check background subsystems).
- `alpha` was fit under different conditions than the current workload. Run a
  new calibration block.
- The generation layer is not respecting `max_tokens`. Verify that
  `response_len == len(actual_response)` before recording.

### The calibration log grows without bound
Every `record_turn` and every calibration/verification event writes a line.
That's intentional (it's the audit trail), but you should rotate it. Either
point `calibration_log` at a rotated path per shift or truncate offline once
you've extracted what you need.

## Local MoE

### `Router executed more than one expert` in `verify_moe_pipeline`
`LocalMoERouter.route(vector, k=1)` should activate exactly one expert. If
this trips, something has replaced `_softmax` or the top-k selection. The
canonical implementation uses `np.argsort(probabilities[0])[-k:][::-1]`.

### `ValueError: input_vector must be shaped (batch, 4)`
Router inputs are 2-D `(batch, dimension)`. The default `dimension=4`. Use
`np.atleast_2d(vector)` or `vector[None, :]` before calling `route`.

### `speculative_lookahead_tau(alpha=1.0, gamma=γ)`
Returns `γ + 1`. The formula `(1 - alpha^(γ+1)) / (1 - alpha)` is undefined
for `alpha == 1`; the function handles that case explicitly. Values of
`alpha` are clipped into `[0, 1]`.

## Data conventions

### `retrieval_count` returns `0` for a memory that has entries
The function accepts several fallback keys (`retrieval_count`,
`retrieved_count`, `access_count`, `times_retrieved`). If none are present or
none are integer-convertible, it returns 0. Check the metadata keys you
actually persist.

### `as_timestamp` returns `None`
Passed value wasn't a number, ISO-8601 string, or numeric string. Empty
strings, `None`, and unparseable formats all return `None`. The report
buckets missing timestamps into `unknown`.

### `load_jsonl_records` returns an empty list
The file was empty. If the file starts with `[` or `{`, the loader treats it
as a JSON blob and returns either the list, `records`, `turns`, or a single
`[obj]` wrapper. Otherwise every non-empty line is parsed as JSON. Mixed
JSON + JSONL in the same file is not supported.

## Tests and self-checks

### `python3 core/cognitive_governor.py` fails on `alpha` bounds
The self-check asserts the fitted `alpha` lands in `[1.5e-5, 2.5e-5]` for
its deterministic seed. If this ever fails, either the true alpha in the
synthetic loop was changed or the noise sigma was raised. Check for edits
to `_self_check()` before assuming a regression.

### `verify_moe_pipeline` says "Lookahead did not improve expected verifier throughput"
The check requires `speculative_lookahead_tau(0.8, 4) > 3.0`. That formula is
purely arithmetic; if it fails, the τ implementation has drifted. Restore
the closed-form `(1 - alpha^(gamma + 1)) / (1 - alpha)`.

### Tests import `psc_scaled` from top-level
`tests/test_psc_scaled.py` uses `from psc_scaled import ...` — that works
because `pytest` adds the repository root to `sys.path`. Run tests from the
repo root, not from inside `tests/`.
