# Scripts Runbook

CLI wrappers around the `core/` and `psc_scaled.py` modules live under
`scripts/`. This page catalogs each one: what it does, how to call it, and
the shape of its output. Everything below is verified against the current
tree with `python3 3.10+` and `numpy 2.x`.

Each script inserts the repository root into `sys.path` at import time, so
you can run them directly from a fresh checkout — no `pip install .` step.

## `scripts/psc_scaled_validation.py`

End-to-end validation for the PSC scaled engine
(see [`psc_engine.md`](psc_engine.md)). Runs six synthetic trajectories
(`T1 Smooth`, `T2 Spike`, `T3 Regime`, `T4 Feedback`, `T5 Noisy`, `T6 Compound`),
prints per-trajectory lead time, miss rate, false-positive rate, MAE, plus
chaos and horizon distributions at alert time. Also runs a scale benchmark
for `[16, 50, 100, 200, 500]` dimensions.

```bash
python3 scripts/psc_scaled_validation.py --no-figure
```

Flags:

- `--no-figure` — skip rendering the four-panel matplotlib summary.
- `--output PATH` — path for the rendered PNG (default: `outputs/psc_scaled_validation.png`).

Only `--output` needs matplotlib. Without `--no-figure` the script requires
`matplotlib`; with it, only NumPy is needed.

## `scripts/alpha_calibration_run.py`

Runs the linear-fit calibration for `delta_energy = alpha * response_len`.
Either against a real JSONL trajectory or against a 50-turn synthetic gated
trajectory with a known injected `alpha` (`2.5e-5`) and Gaussian noise.

```bash
# Synthetic trajectory (default 50 turns, seed 42)
python3 scripts/alpha_calibration_run.py

# Real trajectory JSONL with prev_energy/new_energy/response_len fields
python3 scripts/alpha_calibration_run.py --trajectory-jsonl logs/turns.jsonl
```

Flags:

- `--turns N` — synthetic turn count (default 50; must be ≥ 20).
- `--seed N` — RNG seed (default 42).
- `--output PATH` — optional JSONL dump of the synthetic records.
- `--trajectory-jsonl PATH` — use a real trajectory instead of synthetic data.
  Records need at least `response_len` and either `delta_energy` or both
  `prev_energy` and `new_energy`.

Exits non-zero if fewer than 20 clean records are available. Output prints the
`AlphaCalibrationResult` (samples, alpha, correlation, r_squared, verdict).

## `scripts/cognitive_governor_measurements.py`

Runs an 80-turn synthetic simulation (30 ungated turns for calibration, 50
gated turns under stress) to measure the closed-loop effect of the token
gate. Prints the full `GovernorMeasurements` plus counterfactual gate savings
computed against a hypothetical ungated stress run.

```bash
python3 scripts/cognitive_governor_measurements.py --turns 80
```

Flags:

- `--seed N` — RNG seed (default 42).
- `--turns N` — total turns (default 80; must be ≥ 31 to include the 30-turn
  calibration block).

The script fails fast if `--turns <= 30`.

## `scripts/memory_lifecycle_query.py`

Reports memory-lifecycle health from ChromaDB metadata (staleness, missing
`last_accessed`, frequently-retrieved-only-in-poor-sessions). Also supports an
offline JSON fixture path so you can run it without a live Chroma
installation.

```bash
# Live Chroma:
python3 scripts/memory_lifecycle_query.py --path data/chroma --collection drift_memory

# Offline JSON fixture:
python3 scripts/memory_lifecycle_query.py --metadata-json tests/fixtures/memory_metadatas.json
```

Flags:

- `--path PATH` — ChromaDB persistent path (default `data/chroma`).
- `--collection NAME` — Chroma collection (default `drift_memory`).
- `--stale-days N` — days-since-`last_accessed` threshold (default 30).
- `--frequent-threshold N` — retrieval-count threshold for "frequent" memories
  (default 5).
- `--metadata-json PATH` — JSON fixture with either a top-level list or
  `{"metadatas": [...]}`.

The Chroma path calls `chromadb.PersistentClient(path=...).get_collection(name).get(include=["metadatas"])`.
The script exits with a helpful message if `chromadb` is not installed and no
`--metadata-json` is provided.

## `scripts/retrieval_evaluation.py`

Repeat-frequency and age-bucket report over the same memory metadata format
as the lifecycle query.

```bash
python3 scripts/retrieval_evaluation.py --metadata-json tests/fixtures/memory_metadatas.json
```

Flags:

- `--path PATH`, `--collection NAME`, `--metadata-json PATH` — same as the
  lifecycle query.
- `--repeat-threshold N` — retrieval count that flags a memory as "repeated"
  (default 5, must be ≥ 1).

The output prints repeated-memory count and retrieved ratio, then the full
`build_retrieval_report` dict. Age buckets: `0_7_days`, `8_30_days`,
`31_90_days`, `over_90_days`, `unknown`.

## `scripts/roi_dashboard.py`

Per-subsystem ROI aggregation from trajectory records (JSON array, single
JSON object with `records`/`turns`, or JSONL — the loader detects the shape).

```bash
python3 scripts/roi_dashboard.py tests/fixtures/roi_records.jsonl
```

Prints a table (`Subsystem`, `Cost ms`, `Benefit`, `ROI/ms`, `Label`) followed
by the JSON. `Label` values: `high` (≥0.70), `medium` (≥0.35), `low` (>0),
`none` (=0), `unknown` (no benefit signal).

Trajectory records may use any of these shapes:

```json
{"subsystems": {"DMU": {"cost_ms": 25, "benefit": 0.82}}}
{"subsystem_costs_ms": {"DMU": 28}, "subsystem_benefits": {"DMU": 0.78}}
{"subsystem": "SelfModify", "cost_ms": 100, "benefit_score": 0.10}
```

`build_roi_dashboard` uses these keys with the fallbacks:

- cost: `cost_ms` → `latency_ms`.
- benefit: `benefit` → `benefit_score`.

## `scripts/local_moe_verify.py`

Exercises the local sparse MoE router and speculative-lookahead τ. In its
default configuration (`--experts 4 --k 1`, no `--use-dii-vector`), it also
runs the `verify_moe_pipeline()` self-check and exits non-zero on failure.

```bash
# Default (self-check + JSON report)
python3 scripts/local_moe_verify.py

# Route on a live DII summary instead of the fixed test vector
python3 scripts/local_moe_verify.py --use-dii-vector

# Sweep more experts / different top-k
python3 scripts/local_moe_verify.py --experts 8 --k 2
```

Flags:

- `--experts N` — number of local experts (default 4).
- `--k N` — top-k activation count (default 1).
- `--seed N` — deterministic seed for gate + expert weights (default 7).
- `--acceptance-rate F` — `alpha` for the speculative-lookahead τ formula
  (default 0.8).
- `--gamma N` — draft length for τ (default 4).
- `--use-dii-vector` — build the router input from a synthetic DII summary
  via `vector_from_dii_summary` instead of the fixed test vector.

Output JSON includes `chosen_experts`, `chosen_names`, `router_probabilities`,
`active_expert_count`, `execution_counts` (proves only top-k experts ran),
and `speculative_tau`.

## Log locations

Scripts write structured logs into (or below) the caller's working
directory. Notable defaults:

- `CognitiveGovernor(calibration_log=...)` — default
  `logs/governor_calibration.jsonl`. Parent directories are created eagerly.
- `scripts/psc_scaled_validation.py` figure — default
  `outputs/psc_scaled_validation.png`.
- `scripts/alpha_calibration_run.py` internally uses a `tempfile.mkstemp`
  path for its governor log; the `--output` flag writes the synthetic
  trajectory as JSONL to a caller-chosen path.

`outputs/` is `.gitignore`d.
