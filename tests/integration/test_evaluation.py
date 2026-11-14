import json
import pytest
from airline_market.experiments.information_value import toy_config
from airline_market.eval import ExperimentRunner, SeedPlan
from airline_market.eval.reports import paired_statistics


def test_generic_legacy_quick_and_workers(tmp_path):
    config = toy_config(scenarios=1, legacy=True)
    config["seeds"] = {
        "train": [11, 2],
        "validation": [],
        "test": [31, 2],
        "planning_master": 773,
    }
    result = ExperimentRunner(config).run(tmp_path / "run")
    old = json.load(open("tests/golden/legacy_quick.json"))
    for m, rows in result["episodes"].items():
        for a, b in zip(rows, old["episodes"][m]):
            for metric in old["summary"][m]:
                assert a[metric] == b[metric]
    assert (tmp_path / "run/report.md").exists() and (
        tmp_path / "manifest.jsonl"
    ).exists()
    config["workers"] = 4
    parallel = ExperimentRunner(config).run()
    assert result["summary"] == parallel["summary"]
    assert result["episodes"] == parallel["episodes"]
    with pytest.raises(ValueError):
        SeedPlan((1, 2), (), (2, 3))
    with pytest.raises(ValueError):
        paired_statistics([{"seed": 1}], [{"seed": 2}], "profit")
