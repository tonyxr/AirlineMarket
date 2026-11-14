import json
import pytest
from pathlib import Path
from airline_market.legacy_toy import Config
from airline_market.legacy_toy.experiment import run_experiment


def assert_diagnostics_close(actual, expected):
    """Allow floating diagnostic roundoff without weakening decision identity."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_diagnostics_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_diagnostics_close(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_legacy_quick_golden():
    actual = run_experiment(
        Config(), [11, 12], [31, 32], scenarios=1, trace=True, progress_every=0
    )
    golden = json.loads(
        (Path(__file__).parents[1] / "golden/legacy_quick.json").read_text()
    )
    # JSON canonicalizes tuple records to lists.
    actual = json.loads(json.dumps(actual))
    # Actions/outcomes stay exact; diagnostics may differ by an ulp across NumPy.
    diagnostics = (
        "belief_entropy",
        "belief_mean_rival_inventory",
        "forecast_objective",
    )
    for method, episodes in actual["episodes"].items():
        for row, expected in zip(episodes, golden["episodes"][method], strict=True):
            for decision, reference in zip(
                row["decisions"], expected["decisions"], strict=True
            ):
                for name in diagnostics:
                    value, target = decision.pop(name), reference.pop(name)
                    assert (
                        value == pytest.approx(target, rel=1e-12, abs=1e-12)
                        if target is not None
                        else value is None
                    )
            assert row == expected
    assert_diagnostics_close(actual["summary"], golden["summary"])
    assert_diagnostics_close(actual["contrasts"], golden["contrasts"])
