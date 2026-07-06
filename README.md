# CREX

CREX is the toolkit backing the **DRIFT** stability-control experiments: a set of
small, testable Python modules that combine into a closed-loop cognitive
governor. It watches a bounded state (energy, focus, coherence, ...), forecasts
degradation with a vectorized PSC engine, gates behavior with an energy-coupled
governor, and reports whether the interventions actually changed the outcome.

Everything is CPU-only, uses NumPy as the only hard runtime dependency, and is
covered by unit tests.

## Subsystems at a glance

| Module | Role | Docs |
| --- | --- | --- |
| `psc_scaled.py` | Batch PSC engine: chaos scoring, dynamic prediction horizons, residual-based confidence, alert gating for up to hundreds of dimensions. | [`docs/psc_engine.md`](docs/psc_engine.md) |
| `core/cognitive_governor.py` | Energy-coupled token gate with modes, alpha calibration, and closed-loop intervention verification. | [`docs/cognitive_governor.md`](docs/cognitive_governor.md) |
| `core/alpha_calibration.py` | Standalone linear fit for `delta_energy = alpha * response_len`. | [`docs/cognitive_governor.md`](docs/cognitive_governor.md) |
| `core/dii_tracker.py` | Dynamic Integration Index sensor with an explicit startup heartbeat. | [`docs/architecture.md`](docs/architecture.md) |
| `core/local_moe.py` | Local sparse Mixture-of-Experts router and speculative-lookahead τ. | [`docs/architecture.md`](docs/architecture.md) |
| `core/failure_taxonomy.py` | Structured failure classification for turn/trajectory logs. | [`docs/architecture.md`](docs/architecture.md) |
| `core/retrieval_evaluation.py` | Repeat-frequency and age-distribution metrics over memory metadata. | [`docs/architecture.md`](docs/architecture.md) |
| `core/roi_dashboard.py` | Per-subsystem cost / benefit / ROI aggregation from trajectory records. | [`docs/architecture.md`](docs/architecture.md) |
| `token_upgrade.py` | Lossless-ish token compression for LLM context payloads. | [`docs/architecture.md`](docs/architecture.md) |

Operational scripts under `scripts/` wrap those modules as CLI tools. See
[`docs/scripts.md`](docs/scripts.md) for a runbook.

## Quickstart

Python 3.10+ and NumPy are required. Install into a virtualenv:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install numpy pytest
# matplotlib is only needed if you want the PSC validation figure
pip install matplotlib
```

Run the tests to prove the tree is healthy:

```bash
python3 -m pytest tests/ -q
```

Run the built-in self-checks for the two heaviest subsystems:

```bash
python3 core/cognitive_governor.py       # governor gate + calibration self-check
python3 -c "from core.local_moe import verify_moe_pipeline; verify_moe_pipeline()"
```

Run a PSC scaled-engine validation sweep (add `--no-figure` to skip matplotlib):

```bash
python3 scripts/psc_scaled_validation.py --no-figure
```

## Project layout

```
psc_scaled.py                 # vectorized PSC batch engine
token_upgrade.py              # LLM prompt/token compression helper
core/
  cognitive_governor.py       # energy-coupled behavioral gate
  alpha_calibration.py        # linear energy-drain fit
  dii_tracker.py              # DII sensor + startup heartbeat
  local_moe.py                # sparse local MoE router
  failure_taxonomy.py         # failure classification for trajectories
  retrieval_evaluation.py     # memory retrieval metrics
  roi_dashboard.py            # per-subsystem ROI aggregation
  sensor_initialization.py    # sensor wake hooks
scripts/                      # CLI wrappers over core modules
tests/                        # unit tests + fixtures
docs/                         # long-form documentation
```

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — how the subsystems compose end-to-end.
- [`docs/psc_engine.md`](docs/psc_engine.md) — PSC scaled engine internals and tuning.
- [`docs/cognitive_governor.md`](docs/cognitive_governor.md) — energy modes, alpha calibration, verification.
- [`docs/scripts.md`](docs/scripts.md) — operational runbook for CLI scripts.
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common pitfalls and fixes.
