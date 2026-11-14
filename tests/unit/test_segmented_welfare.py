"""A.10/B0/V9: closed-form conditional CS and lossless native-request strata."""

import importlib.util
import math
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim.arrivals import Request
from airline_market.sim.choice import systematic_utilities
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("SEGMENTED_WELFARE_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_segmented_welfare", os.environ["SEGMENTED_WELFARE_SOURCE"]
    )
    module = importlib.util.module_from_spec(s)
    sys.modules[s.name] = module
    s.loader.exec_module(module)
else:
    from airline_market.sim import segmented_welfare as module


def fixture():
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 20,
                "capacity_support": [3, 4],
                "arrival_probability": 0.741,
            }
        )
    )
    e = MarketEngine(c)
    e.reset(
        tape=ExogenousTape.generate(
            c, 27, registry=StreamRegistry(25974, "unit/V9_welfare")
        )
    )
    return e


def actions(e):
    return {
        x.code: PriceAction(
            {j: 200 if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def step(e, requests=(), choices=None):
    return e.step(actions(e), _requests=list(requests), _choices=choices)


def labels(e):
    return {
        m.id: ("connecting" if i == 2 else "local")
        for i, m in enumerate(e.compiled.spec.markets)
    }


def summary(e, **kw):
    return module.ordinary_welfare_summary(
        e,
        market_groups=kw.get("market_groups", labels(e)),
        early_stop=kw.get("early_stop", 2),
    )


def test_static_binary_logit_matches_analytic_closed_form():
    assert module.conditional_logit_surplus((math.log(3), 0), 0.5) == pytest.approx(
        math.log(4) / 0.5, abs=1e-14
    )
    assert module.conditional_logit_surplus((0, 0), 2) == pytest.approx(
        math.log(2) / 2, abs=1e-14
    )


def test_outside_normalization_is_shift_invariant():
    base = module.conditional_logit_surplus((1, 2, 0), 0.25)
    assert module.conditional_logit_surplus((1001, 1002, 1000), 0.25) == pytest.approx(
        base, abs=1e-12
    )
    assert module.conditional_logit_surplus(
        (0, 1, 2), 0.25, outside_index=0
    ) == pytest.approx(base, abs=1e-12)


def test_unavailable_and_all_outside_choice_sets():
    assert module.conditional_logit_surplus(
        (-math.inf, math.log(3), 0), 1
    ) == pytest.approx(math.log(4))
    assert module.conditional_logit_surplus((-math.inf, -math.inf, 0), 1) == 0
    assert module.conditional_logit_surplus((0,), 1) == 0


def test_large_utility_does_not_exponentiate_raw_value():
    assert module.conditional_logit_surplus((1000, 0), 0.5) == 2000


def test_finite_class_mixture_integrates_class_specific_alpha():
    rows = ((0.25, (math.log(3), 0), 0.5), (0.75, (0, 0), 2))
    expected = 0.25 * math.log(4) / 0.5 + 0.75 * math.log(2) / 2
    assert module.conditional_mixture_surplus(rows) == pytest.approx(
        expected, abs=1e-14
    )
    assert expected != pytest.approx(
        (0.25 * math.log(4) + 0.75 * math.log(2)) / (0.25 * 0.5 + 0.75 * 2)
    )


@pytest.mark.parametrize("alpha", [0, -1, math.inf, math.nan])
def test_invalid_price_sensitivity_is_not_replaced(alpha):
    with pytest.raises(ValueError):
        module.conditional_logit_surplus((1, 0), alpha)


@pytest.mark.parametrize("u", [(math.nan, 0), (math.inf, 0), (1, -math.inf), ()])
def test_invalid_utilities_or_outside_are_explicit(u):
    with pytest.raises(ValueError):
        module.conditional_logit_surplus(u, 1)


@pytest.mark.parametrize("weights", [(0.2, 0.2), (-0.1, 1.1), (math.nan, 1)])
def test_mixture_mass_is_not_silently_normalized(weights):
    with pytest.raises(ValueError):
        module.conditional_mixture_surplus(tuple((w, (0, 0), 1) for w in weights))


def test_native_logged_logsum_matches_static_closed_form():
    e = fixture()
    c = e.compiled
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    row = e.world.request_log[0]
    taste = c.spec.segments[0].classes[0]
    u = systematic_utilities(
        c, 0, taste, e.trace()[-1]["prices"], np.ones(len(c.spec.products), dtype=bool)
    )
    expected = math.log(1 + math.fsum(math.exp(x) for x in u[:-1])) / taste.alpha
    assert row["consumer_surplus"] == pytest.approx(expected, abs=1e-12)
    assert module.conditional_logit_surplus(u, taste.alpha) == pytest.approx(
        row["consumer_surplus"], abs=1e-12
    )


def test_joint_strata_preserve_all_ordinary_requests_and_sums():
    e = fixture()
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    step(e, [Request(0, 2, 1, 0, 0, 0)], [None])
    step(e, [Request(0, 2, 0, 0, 0, 0)], [None])
    out = summary(e)
    assert (
        out["ordinary"]["logged_requests"] == 3
        and out["ordinary"]["outside_choices"] == 3
    )
    assert (
        sum(x["logged_requests"] for x in out["joint_strata"]) == 3
        and len(out["joint_strata"]) == 16
    )
    assert math.fsum(
        x["conditional_consumer_surplus"] for x in out["joint_strata"]
    ) == pytest.approx(out["ordinary"]["conditional_consumer_surplus"])
    assert {x["segment"] for x in out["by_segment"]} == {"business", "leisure"}
    observed = {
        (x["segment"], x["request_market_group"], x["booking_period_group"]): x[
            "logged_requests"
        ]
        for x in out["joint_strata"]
        if x["logged_requests"]
    }
    assert observed == {
        ("leisure", "local", "early"): 1,
        ("business", "connecting", "early"): 1,
        ("leisure", "connecting", "late"): 1,
    }


def test_empty_registered_strata_are_logged_zero_not_missing_exposure_claim():
    e = fixture()
    out = summary(e)
    assert (
        out["ordinary"]["logged_requests"] == 0
        and out["ordinary"]["conditional_consumer_surplus"] == 0
    )
    assert out["observed_native_periods"] == 0 and len(out["joint_strata"]) == 16
    assert (
        out["total_lifecycle_surplus"] is None
        and out["stranded_lifecycle_consumer_surplus"] is None
    )


def test_missing_labels_preserve_unclassified_and_all_zero_cells():
    e = fixture()
    step(e, [Request(0, 2, 1, 0, 0, 0)], [None])
    out = summary(e, market_groups={})
    assert (
        not out["labels_complete"]
        and sum(
            x["logged_requests"]
            for x in out["joint_strata"]
            if x["request_market_group"] == "unclassified"
        )
        == 1
    )
    assert len(out["by_market"]) == 3 and len(out["joint_strata"]) == 16


def test_market_label_does_not_depend_on_purchase_or_outside():
    e = fixture()
    j = max(
        e.compiled.products_of[0], key=lambda j: len(e.compiled.spec.products[j].legs)
    )
    m = int(e.compiled.product_market[j])
    step(e, [Request(0, m, 0, 0, 0, 0)], [j])
    step(e, [Request(0, m, 0, 0, 0, 0)], [None])
    out = summary(e, market_groups={e.compiled.spec.markets[m].id: "mixed"})
    assert (
        sum(
            x["logged_requests"]
            for x in out["joint_strata"]
            if x["request_market_group"] == "mixed"
        )
        == 2
    )
    assert sum(x["outside_choices"] for x in out["joint_strata"]) == 1


def test_realized_negative_values_are_preserved_and_native_profit_is_dollars():
    e = fixture()
    j = e.compiled.products_of[0][0]
    m = int(e.compiled.product_market[j])
    step(e, [Request(0, m, 0, 0, 0, 0)], [j])
    e.world.request_log[0]["realized_surplus"] = -12.5
    out = summary(e)
    assert out["ordinary"]["realized_surplus"] == -12.5
    expected = e.world.carriers[0].cash - e.compiled.product_cost[j]
    assert out["carrier_profits"]["A"] == expected
    for row in out["ordinary_CS_plus_lambda_profit_index"]:
        assert row["value"] == pytest.approx(
            out["ordinary"]["conditional_consumer_surplus"]
            + row["lambda"] * sum(out["carrier_profits"].values())
        )
    assert (
        out["total_lifecycle_surplus"] is None and not out["primary_B4_welfare_adopted"]
    )


def test_outside_with_unavailable_option_is_not_claimed_denied_demand():
    e = fixture()
    for s in e.world.carriers:
        s.capacity[:] = 0
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    out = summary(e)
    assert (
        out["ordinary"]["outside_with_unavailable_option"] == 1
        and out["denied_requests"] is None
        and out["misallocation"] is None
    )


def test_reentry_histories_are_not_summed_as_extra_welfare_people():
    e = fixture()
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    base = summary(e)
    proxy = SimpleNamespace(
        world=e.world,
        compiled=e.compiled,
        _passengers={7: object()},
        rebooking_history=[{"consumer_surplus": 1000}] * 5,
    )
    out = module.ordinary_welfare_summary(proxy, market_groups=labels(e), early_stop=2)
    assert (
        out["ordinary"] == base["ordinary"] and out["stranded_lineages_observed"] == 1
    )
    assert (
        out["stranded_lifecycle_consumer_surplus"] is None
        and out["total_lifecycle_surplus"] is None
    )


def test_summary_is_read_only_and_calls_no_actors_RNG_or_tape(monkeypatch):
    e = fixture()
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    before = pickle.dumps(e.snapshot())

    def fail(*a, **kw):
        raise AssertionError("unexpected actor/RNG/kernel call")

    monkeypatch.setattr(StreamRegistry, "generator", fail)
    for ctrl in e.controllers.values():
        monkeypatch.setattr(ctrl, "act", fail)

    class AnalystView:
        world = e.world
        compiled = e.compiled

        @property
        def tape(self):
            raise AssertionError("future tape read")

    out = module.ordinary_welfare_summary(
        AnalystView(), market_groups=labels(e), early_stop=2
    )
    assert (
        out["ordinary"]["logged_requests"] == 1 and pickle.dumps(e.snapshot()) == before
    )


@pytest.mark.parametrize("boundary", [-1, 6, True, 1.5])
def test_early_late_window_is_explicit_native_boundary(boundary):
    e = fixture()
    with pytest.raises(ValueError):
        summary(e, early_stop=boundary)


def test_future_or_unknown_rows_are_not_zero_filled_or_silently_dropped():
    e = fixture()
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    e.world.request_log[0]["period"] = 4
    with pytest.raises(ValueError, match="completed native prefix"):
        summary(e)


def test_missing_amount_and_bad_label_are_explicit():
    e = fixture()
    step(e, [Request(0, 0, 0, 0, 0, 0)], [None])
    e.world.request_log[0]["consumer_surplus"] = math.nan
    with pytest.raises(ValueError, match="finite logged"):
        summary(e)
    with pytest.raises(ValueError, match="known O&D"):
        summary(e, market_groups={"bad": "local"})


def test_actual_dated_trigger_rebooking_is_not_extra_ordinary_CS():
    spec = importlib.util.spec_from_file_location(
        "welfare_native_fixture", "tests/unit/test_triggered_dated.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    e, reports = native.fixture()
    b = native.strand(e)
    e.present_cancelled_itinerary(b)
    native.empty(e, reports)
    assert len(e.world.request_log) == 1 and len(e.rebooking_history) == 1
    before = pickle.dumps(e.snapshot())
    out = summary(e)
    assert (
        out["ordinary"]["logged_requests"] == 1
        and out["stranded_lineages_observed"] == 1
    )
    assert (
        out["ordinary"]["conditional_consumer_surplus"]
        == e.world.request_log[0]["consumer_surplus"]
    )
    assert (
        out["stranded_lifecycle_consumer_surplus"] is None
        and out["total_lifecycle_surplus"] is None
    )
    assert e.world.carriers[0].revenue == 80 and e.world.carriers[1].refunds == 300
    assert pickle.dumps(e.snapshot()) == before


def test_explicit_unclassified_labels_are_not_claimed_complete():
    e = fixture()
    out = summary(
        e, market_groups={m.id: "unclassified" for m in e.compiled.spec.markets}
    )
    assert not out["labels_complete"] and len(out["joint_strata"]) == 16


def test_empty_mixture_and_outside_index_cannot_be_invented():
    with pytest.raises(ValueError):
        module.conditional_mixture_surplus(())
    with pytest.raises(ValueError):
        module.conditional_logit_surplus((1, 0), 1, outside_index=2)
