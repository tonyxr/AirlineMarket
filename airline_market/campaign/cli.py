"""Campaign run/status/aggregate commands (validation plan R0.4)."""

import argparse
import json
from pathlib import Path

from .design import load_campaign
from .runner import aggregate, campaign_status, run_campaign


def main(argv=None) -> None:
    """Execute or inspect a declared campaign without launching duplicate jobs."""
    parser = argparse.ArgumentParser(prog="airline-market campaign")
    parser.add_argument("operation", choices=["run", "status", "aggregate", "expand"])
    parser.add_argument("path")
    parser.add_argument("--out")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "run":
        result = run_campaign(args.path, args.out, retry_failed=args.retry_failed)
    elif args.operation == "expand":
        from .design import expand_campaign

        result = expand_campaign(load_campaign(args.path))
    else:
        path = Path(args.path)
        if path.is_file():
            value = load_campaign(path)
            if "out" not in value:
                parser.error(
                    "use a run directory or a config with an explicit out path"
                )
            path = Path(value["out"])
        result = (
            campaign_status(path) if args.operation == "status" else aggregate(path)
        )
    print(json.dumps(result, indent=2, allow_nan=False))
