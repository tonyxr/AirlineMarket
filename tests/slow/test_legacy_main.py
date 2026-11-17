import json

import pytest

from airline_market.eval import ExperimentRunner
from airline_market.experiments.information_value import toy_config
from tests.legacy.test_golden import assert_diagnostics_close


@pytest.mark.slow
def test_legacy_main_generic_reproduction(tmp_path):
    cfg = toy_config(scenarios=32, legacy=True, workers=4)
    result = ExperimentRunner(cfg).run(tmp_path / "generic_legacy_main")
    with open("tests/golden/legacy_main.json") as stream:
        expected = json.load(stream)
    for method, stats in expected["summary"].items():
        for metric, s in stats.items():
            for key in ("mean", "standard_error", "ci95_normal"):
                assert_diagnostics_close(result["summary"][method][metric][key], s[key])
    for contrast, stats in expected["contrasts"].items():
        for metric, s in stats.items():
            assert_diagnostics_close(
                result["contrasts"][contrast][metric]["mean"], s["mean"]
            )
    for method, rows in expected["episodes"].items():
        for actual, target in zip(result["episodes"][method], rows, strict=True):
            for metric in expected["summary"][method]:
                assert actual[metric] == target[metric]
