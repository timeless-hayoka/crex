# CREX / DRIFT Architecture

CREX is a collection of small, independently-testable modules that combine into
a closed-loop stability control system. This page describes how they fit
together and what data flows between them. Deep dives for the two heaviest
subsystems live in [`psc_engine.md`](psc_engine.md) and
[`cognitive_governor.md`](cognitive_governor.md).

## Design goals

- **CPU-only, NumPy-only** runtime — no GPU, no ML framework at inference time.
- **Vectorized batch math** — the PSC engine processes all monitored dimensions
  in one batch per cycle so 16→500 dimensions cost roughly the same.
- **Falsifiable, closed-loop measurements** — every intervention (token gate,
  alert) writes structured logs so its effect can be measured after the fact.
- **Deterministic self-checks** — every heavy module has a `__main__` block or a
  `verify_*` function that asserts the contract without external data.

## End-to-end loop

The intended per-turn flow is:

1. **Sense** — `core.sensor_initialization.initialize_sensors()` runs once at
   startup so the DII sensor is awake. Each turn, `DIITracker.update_from_interaction(text)`
   scores the new interaction and updates a variance-bearing history.
2. **Forecast** — a `PSCBatchEngine` (see [`psc_engine.md`](psc_engine.md))
   accepts a normalized state vector (`{dimension: value in [0, 1]}`) via
   `push_state` and returns a `PSCBatchResult` from `run()` with per-dimension
   predictions, chaos scores, dynamic horizons, residual-based confidence, and
   new/active alerts.
3. **Route** — `core.local_moe.LocalMoERouter` picks top-`k` local experts for a
   4-D control vector (typically derived from the DII summary via
   `vector_from_dii_summary`). Only the chosen experts actually execute, and
   `speculative_lookahead_tau(alpha, gamma)` gives the expected accepted-token
   count for a speculative-decoding pass.
4. **Gate** — `core.cognitive_governor.CognitiveGovernor.apply(current_energy, turn)`
   maps energy to one of five modes (`NORMAL`, `MODERATE`, `LOW_POWER`,
   `CRITICAL`, `HOLD`) and returns a token budget plus an audit trail of
   interventions. See [`cognitive_governor.md`](cognitive_governor.md).
5. **Act & record** — the caller generates a response bounded by
   `GateResult.max_tokens`, observes the new energy, and calls
   `record_turn(prev_energy, new_energy, response_len, gate_result)`. The
   governor appends a JSONL calibration record and, once `_fitted_alpha` is
   known, closed-loop verifies whether the gate actually reduced drain.
6. **Observe** — offline scripts under `scripts/` roll the JSONL logs into:
   - alpha fits (`core.alpha_calibration.fit_alpha`),
   - retrieval health (`core.retrieval_evaluation.build_retrieval_report`),
   - memory lifecycle (`scripts/memory_lifecycle_query.py`),
   - subsystem ROI (`core.roi_dashboard.build_roi_dashboard`),
   - failure taxonomies (`core.failure_taxonomy.summarize_failure_types`).

The loop is deliberately linear so any single stage can be exercised in tests
with synthetic inputs.

## Module responsibilities

### `psc_scaled.py`
Vectorized PSC (Projection–Stability–Confidence) engine. See
[`psc_engine.md`](psc_engine.md) for the internals. Public surface:

- `PSCStateBuffer` — fixed-size circular buffer over a set of named dimensions.
- `PSCBatchEngine(dimensions, policy=..., buffer_size=..., min_history=...)` —
  push normalized state vectors, get a `PSCBatchResult`.
- `_batch_chaos_score`, `_batch_residual_confidence`, `_dynamic_n_steps` —
  building blocks re-used by the validation sweeps and unit tests.
- `benchmark_scale(dim_counts, n_cycles, seed)` — synthetic latency benchmark.

### `core/cognitive_governor.py`
Energy → behavior controller. Key types:

- `CognitiveGovernor(calibration_log=..., energy_min=0.10)` — stateful, thread-safe.
- `GateResult` — one decision (mode, max tokens, interventions, energy snapshot).
- `TurnRecord` — one observed turn (prev/new energy, delta, response length).
- `GovernorMeasurements` — aggregated closed-loop measurements.

### `core/alpha_calibration.py`
Pure-function fit for the linear no-intercept model
`delta_energy = alpha * response_len`. Used both by the governor's internal
`calibrate_alpha()` and by the offline `scripts/alpha_calibration_run.py`.

### `core/dii_tracker.py`
Dynamic Integration Index sensor. Deterministic per-text scoring based on unique
word ratio, length, and punctuation structure. Exposes:

- `update_from_interaction(text)` → new reading (0..1).
- `get_current()`, `is_awake()`, `variance()`, `summary()`, `reset()`.
- Module-level singleton via `get_dii_tracker()`.

### `core/local_moe.py`
Local sparse MoE with deterministic per-expert weights. `LocalMoERouter.route`
computes a softmax over gate scores, keeps only the top-`k` experts, and
executes _only_ those (verified by `execution_counts()`).
`speculative_lookahead_tau(alpha, gamma)` returns
`(1 - alpha^(gamma+1)) / (1 - alpha)` (with `alpha=1` treated as `gamma+1`).

### `core/failure_taxonomy.py`
String-pattern classifier that maps raw failure messages onto a fixed
`FailureType` enum (timeout, quota_exceeded, math_hallucination, seam_not_found,
division_by_zero, sensor_stuck, sensor_invalid, retrieval_empty,
retrieval_stale, update_read_path_mismatch, latency_spike, unknown). Also
supports latency-based timeout classification.

### `core/retrieval_evaluation.py`
Metrics over ChromaDB-style memory metadata:

- `retrieval_count(metadata)` — tolerant of `retrieval_count`, `retrieved_count`,
  `access_count`, or `times_retrieved` keys.
- `build_retrieval_report(metadatas, now=..., repeat_threshold=5)` — returns
  `retrieved_memory_ratio`, `repeat_retrieval_ratio`,
  `memory_repeat_frequency`, `memory_age_distribution` (buckets: 0–7, 8–30,
  31–90, >90 days, and unknown).

### `core/roi_dashboard.py`
Consumes trajectory records in three shapes (see fixtures for examples):

1. `{"subsystems": {"NAME": {"cost_ms": ..., "benefit": ...}, ...}}`
2. `{"subsystem_costs_ms": {...}, "subsystem_benefits": {...}}`
3. `{"subsystem": "NAME", "cost_ms": ..., "benefit_score": ...}`

`build_roi_dashboard(records)` returns a per-subsystem list with mean cost,
mean benefit, `roi_per_ms`, and a coarse label (`high`/`medium`/`low`/`none`/
`unknown`).

### `token_upgrade.py`
Prompt/context compressor for LLM inputs. Minifies embedded JSON blocks,
collapses whitespace and comments in fenced JS/CSS/HTML/Python blocks, and
strips a small allow-list of conversational fluff prefixes. Idempotent enough
to run on already-compressed prompts.

Run directly on a file:

```bash
python3 token_upgrade.py path/to/prompt.md
```

## Data conventions

- **Dimension values** are always normalized to `[0, 1]`. `PSCStateBuffer.push`
  clips out-of-range values silently.
- **Polarity** is per-dimension: `+1` means higher is safer (`focus`, `energy`,
  `context_integrity`, ...); `-1` means lower is safer (`threat_pressure`,
  `latency_pressure`, ...). The default map lives in `DIMENSION_POLARITY`.
- **Energy** is a scalar in `[0, 1]`. Values outside that range or non-finite
  values collapse the governor into `HOLD` mode with an 80-token budget.
- **Timestamps** in metadata may be numeric epoch seconds or ISO 8601 strings;
  the loaders (`retrieval_evaluation.as_timestamp`,
  `memory_lifecycle_query._as_timestamp`) accept either.
- **Trajectory logs** are JSONL by convention, but `core.roi_dashboard.load_jsonl_records`
  also accepts a JSON array or a single JSON object with a `records`/`turns`
  key.

## Threading and I/O

- `CognitiveGovernor` and `DIITracker` hold a `threading.Lock` around mutable
  state, so a single instance is safe to share across worker threads.
- The governor's calibration log is opened in append mode per record. There is
  no batching or async flushing, so callers should point `calibration_log` at a
  path they don't mind hitting once per turn.
- The PSC engine is not thread-safe; construct one per stream of state
  vectors.

## Testing hooks

- Every subsystem has a corresponding `tests/test_*.py`.
- `tests/fixtures/memory_metadatas.json` and `tests/fixtures/roi_records.jsonl`
  are the canonical offline fixtures for retrieval/lifecycle/ROI reports.
- Self-checks: `python3 core/cognitive_governor.py`, and
  `from core.local_moe import verify_moe_pipeline; verify_moe_pipeline()`.
