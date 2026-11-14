"""S1 task integration with the actual worker context, on a new micro fixture."""

import gzip
import pickle

import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.worker import CellContext
from airline_market.experiments.screen_study import run
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.spec import dump_spec


def test_s1_entry_uses_native_context_and_retains_policy(tmp_path):
    source = tmp_path / "source.json"
    dump_spec(
        screen_spec(
            {
                "T": 2,
                "actions": 2,
                "n_spokes": 1,
                "price_scale": 1.1,
                "outside_utility_shift": 0.1,
                "arrival_probability": 0.2,
            }
        ),
        source,
    )
    directory = tmp_path / "cell"
    directory.mkdir()
    cell = {
        "parameters": {
            "source_spec": str(source),
            "source_spec_sha256": file_hash(source),
            "arm": "direct",
            "setting_id": "new-context-fixture",
        },
        "limits": {"max_nodes": 100000},
    }
    ctx = CellContext(directory)
    result = run(cell, ctx)
    assert result["status"] == "solved" and result["policy_records"] > 0
    with gzip.open(directory / "optimal_policy.pkl.gz", "rb") as stream:
        node, index = pickle.load(stream)
    assert node[0] in (0, 1) and index in (0, 1)
    assert ctx.publish()["stage"] == "policy_diagnostics"
    assert result["diagnostics"]["revenue_A"] > 0


def test_s1_rejects_changed_source_before_solving(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        run(
            {"parameters": {"source_spec": str(source), "source_spec_sha256": "wrong"}},
            None,
        )
