"""Benchmark execution and opt-in measured regression checks (R0.3)."""

import argparse
import json
from pathlib import Path

from .harness import compare_baseline, run_benchmarks


def main(argv=None) -> None:
    """Run a saved benchmark design or compare two measured baseline files."""
    parser = argparse.ArgumentParser(prog="airline-market bench")
    parser.add_argument("operation", choices=["run", "check"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--out", default="results/R0/bench")
    parser.add_argument("--baseline", default="config/benchmarks/baseline.json")
    args = parser.parse_args(argv)
    if args.operation == "run":
        parameters = json.loads(Path(args.path).read_text()) if args.path else {}
        result = run_benchmarks(parameters, args.out)
        print(
            json.dumps(
                {
                    "output": str(Path(args.out).resolve()),
                    "operations": len(result["benchmarks"]),
                },
                indent=2,
            )
        )
    else:
        failures = compare_baseline(
            json.loads(Path(args.path).read_text()),
            json.loads(Path(args.baseline).read_text()),
        )
        print(json.dumps({"regressions": failures}, indent=2))
        if failures:
            raise SystemExit(1)
