"""Reproduce the disjoint external-load input without repeating the national scan."""

import argparse
import json
from pathlib import Path


def partition(source, out):
    """Verify existing partitions byte-for-byte; write only absent derived files."""
    patterns = json.loads(source.read_text())
    external = [p for p in patterns if p["category"] in
                ("off_pool_od", "in_pool_unselected_market")]
    deferred = [p for p in patterns if p["category"] == "in_pool_unrepresented_path"]
    represented = [p for p in patterns if p["category"] == "represented_core"]
    if len(external) + len(deferred) + len(represented) != len(patterns):
        raise ValueError("unknown coverage category")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("patterns_external.json", external),
                       ("patterns_inside_scope_deferred.json", deferred)]:
        text = json.dumps(rows, indent=2) + "\n"
        path = out / name
        if path.exists():
            if path.read_text() != text:
                raise ValueError("existing partition differs: " + str(path))
        else:
            path.write_text(text)
    return {"external_patterns": len(external), "deferred_patterns": len(deferred),
            "represented_patterns": len(represented), "verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(partition(args.source, args.out))
