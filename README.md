# AirlineMarket

Python airline network simulator and pricing research framework. Physical itineraries consume leg capacity; the implementation includes passenger choice, demand, inventory beliefs, pricing policies, solvers, evaluation and optional learning components.

## Install

Python 3.10 or later is required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
```

Run the local tests with `.venv/bin/python -m pytest -q`. Optional data, study and learning dependencies are declared in `pyproject.toml`. Tests that depend on separate local experiment artifacts need those inputs; this snapshot does not include research result banks.

## Contents

- `airline_market/`: simulator, native policy components, inference, solvers, pipelines and learning code.
- `tests/`: tests, synthetic miniature data fixtures and legacy synthetic golden fixtures.
- `config/`: declared specifications and experiment configurations. Publishing a configuration does not authorize running it.
- `analysis/`: Python analysis tools.
- `scripts/`: tracked project utilities.
- `REUSE.md`: source attribution and adaptations for reused components.

This is a code-only snapshot of local source commit `3c769e20a312041bfe9568466525b96d869ab250`. Original source bytes are preserved. `SOURCE_SNAPSHOT.json` records file hashes. Research results, downloaded datasets, paper drafts, local process state and the private development history are not part of this public repository. Some tools therefore require separately supplied local artifacts. Historical campaigns remain inactive; the snapshot does not claim empirical calibration, full validation readiness or planner adoption.
