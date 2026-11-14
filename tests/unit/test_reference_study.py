"""Reference companion inputs must preserve the compact instance's exact JSON."""

from pathlib import Path

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.experiments.reference_study import solve
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market, dump_spec


def test_saved_family_spec_preserves_metadata_and_action_grid(tmp_path):
    spec = build_instance(
        "hub_feeder_family",
        {
            "n_spokes": 1,
            "T": 2,
            "actions": 2,
            "lag_periods": 0,
            "reservation_gb": 3,
            "connecting_share": 0.7,
        },
    )
    compiled = compile_market(spec)
    path = tmp_path / "source.json"
    dump_spec(spec, path)
    expected = (
        ExactBeliefDP(compiled, "A", "direct", backend="compact", max_nodes=20000)
        .solve()
        .value
    )

    class Context:
        directory = tmp_path / "reference"
        counters = lambda self: {}

    Context.directory.mkdir()
    result = solve(
        {
            "generator": "hub_feeder_family",
            "limits": {"max_nodes": 20000},
            "parameters": {
                "source_spec": str(path),
                "source_spec_sha256": file_hash(path),
                "expected_spec_hash": compiled.spec_hash,
                "expected_compact_value": expected,
                "T": 2,
                "actions": 2,
                "arm": "direct",
                "reservation_gb": 1,
            },
        },
        Context(),
    )
    assert result["status"] == "solved"
    assert result["spec_hash"] == compiled.spec_hash
    assert abs(result["value"] - expected) < 1e-10
