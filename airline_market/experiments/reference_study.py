"""Reference checks of frozen compact specifications, including generator metadata."""

from pathlib import Path

from ..campaign.artifacts import file_hash
from ..campaign.tasks import exact_dp
from ..spec import compile_market, load_spec


def solve(cell, context):
    """Load the exact saved source spec; provenance must not alter model metadata.

    Rebuilding a family instance with expected-value/source-path parameters
    would put those parameters into its metadata and change its spec hash.
    Loading the frozen JSON also preserves its already-restricted action grid.
    """
    parameters = cell["parameters"]
    path = Path(parameters["source_spec"])
    if file_hash(path) != parameters["source_spec_sha256"]:
        raise ValueError("source compact specification changed")
    compiled = compile_market(load_spec(path))
    if compiled.spec_hash != parameters["expected_spec_hash"]:
        raise ValueError("reference/source specification identity mismatch")
    resolved = {
        **cell,
        "generator": str(path),
        "parameters": {
            "arm": parameters["arm"],
            "backend": "reference",
            "focal": compiled.spec.carriers[0].code,
        },
    }
    return exact_dp(resolved, context)
