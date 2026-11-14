"""Closed-form posterior information checks (validation plan Appendix A.4)."""

from math import log
import pytest

from airline_market.belief.info import (
    entropy,
    expected_information_gain,
    mutual_information,
    total_correlation,
)


def test_independence_and_perfect_correlation():
    independent = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}
    correlated = {(0, 0): 0.5, (1, 1): 0.5}
    assert entropy(independent) == pytest.approx(2 * log(2))
    assert mutual_information(independent) == pytest.approx(0, abs=1e-15)
    assert total_correlation(independent) == pytest.approx(0, abs=1e-15)
    assert mutual_information(correlated) == pytest.approx(log(2))
    assert total_correlation(correlated) == pytest.approx(log(2))


def test_information_gain_of_informative_and_ancillary_signal():
    prior = {0: 0.5, 1: 0.5}
    assert expected_information_gain(prior, {0: {0: 1}, 1: {1: 1}}) == pytest.approx(
        log(2)
    )
    assert expected_information_gain(
        prior, {0: {0: 0.4, 1: 0.6}, 1: {0: 0.4, 1: 0.6}}
    ) == pytest.approx(0, abs=1e-15)


def test_invalid_signal_channel_is_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        expected_information_gain({0: 1}, {0: {0: 0.5}})
