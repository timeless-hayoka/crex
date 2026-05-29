# crex

PSC scaled engine validation for DRIFT V4.

## Contents

- `psc_scaled.py` - vectorized batch PSC engine with circular state buffers,
  continuous chaos scoring, dynamic forecast horizons, and residual confidence.
- `scripts/psc_scaled_validation.py` - simulation and scale benchmark for the
  three V4 upgrades.
- `tests/test_psc_scaled.py` - focused standard-library unit tests.

## Quick checks

```bash
python3 -m unittest discover -s tests
python3 scripts/psc_scaled_validation.py --no-figure
```
