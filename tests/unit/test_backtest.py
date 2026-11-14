import pytest
from airline_market.eval.backtest import Backtest


def test_synthetic_pre_post_backtest():
    truth = 2.0
    pre = [truth * x for x in [1.0, 2.0, 3.0]]
    test = Backtest(
        "known demand multiplier",
        (0, 3),
        (3, 5),
        lambda y: sum(y) / 6.0,
        lambda estimate, period: {
            "mean": [estimate * 4, estimate * 5],
            "low": [estimate * 4 - 0.1, estimate * 5 - 0.1],
            "high": [estimate * 4 + 0.1, estimate * 5 + 0.1],
        },
    )
    result = test.run(pre, {"value": [8.0, 10.0], "passengers": [10, 20]})
    assert result["weighted_mse"] == 0.0 and result["coverage"] == 1.0
