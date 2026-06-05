# crex

PSC scaled engine validation for DRIFT V4.

## Contents

- `psc_scaled.py` - vectorized batch PSC engine with circular state buffers,
  continuous chaos scoring, dynamic forecast horizons, and residual confidence.
- `core/cognitive_governor.py` - energy-coupled token gating with calibration
  logs, intervention verification, and aggregate measurements.
- `core/failure_taxonomy.py` - structured failure labels for trajectory
  metadata and raw-log classification.
- `core/retrieval_evaluation.py` - metadata-only retrieval repeat and memory
  age distribution metrics.
- `core/roi_dashboard.py` - subsystem cost/benefit aggregation for
  trajectory-style records.
- `scripts/psc_scaled_validation.py` - simulation and scale benchmark for the
  three V4 upgrades.
- `scripts/cognitive_governor_measurements.py` - synthetic closed-loop
  measurement run for governor calibration and token-gate impact.
- `scripts/memory_lifecycle_query.py` - ChromaDB metadata report for stale,
  missing-access, and poor-session memory lifecycle signals.
- `scripts/retrieval_evaluation.py` - ChromaDB metadata report for retrieval
  repeat frequency and memory age distribution.
- `scripts/roi_dashboard.py` - JSON/JSONL subsystem ROI summary.
- `tests/test_psc_scaled.py` - focused standard-library unit tests.

## Quick checks

```bash
python3 -m unittest discover -s tests
python3 scripts/psc_scaled_validation.py --no-figure
python3 scripts/cognitive_governor_measurements.py
python3 scripts/memory_lifecycle_query.py --metadata-json path/to/metadatas.json
python3 scripts/retrieval_evaluation.py --metadata-json path/to/metadatas.json
python3 scripts/roi_dashboard.py path/to/trajectory_records.jsonl
```
