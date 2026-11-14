"""Local fixtures for outside-network choice, physical identity and fare accounting."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.experiments.empirical_preflight import prepare, measure
from airline_market.pipeline.behavioral_network import initialize_cohort
from airline_market.sim.choice import systematic_utilities, choice_probabilities
from airline_market.sim.revenue_proxy import fit_fee_moments, fee_proxy, net_proxy_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.ledger import book
from airline_market.spec import compile_market, dump_spec


def inputs():
    """A marketed regional resource plus a competing core resource and outside traffic."""
    legs = pd.DataFrame([{"id": c + "_regional", "marketing_carrier": c,
        "cohort_date": "2022-06-15", "origin": "ORD", "dest": "BOS",
        "departure_local": "2022-06-15T09:00:00-05:00",
        "arrival_local": "2022-06-15T12:00:00-04:00", "distance_mi": 867.0,
        "capacity_low": 10, "capacity_high": 12, "probability_high": 0.25}
        for c in ["UA", "AA"]])
    products = pd.DataFrame([{"id": c + "_p", "carrier": c, "market": "ORD-BOS",
        "cohort_date": "2022-06-15", "n_legs": 1, "leg_ids": [c + "_regional"],
        "distance_mi": 867.0, "duration_h": 2.0,
        "departure_local": "2022-06-15T09:00:00-05:00"} for c in ["UA", "AA"]])
    traffic = pd.DataFrame([{"market": "ORD-BOS", "path_network": c,
                            "model_path_eligible": eligible, "passengers": n}
                           for c, eligible, n in [("UA", True, 500), ("AA", True, 300),
                                                   ("DL", False, 150), ("UA", False, 50)]])
    fares = pd.DataFrame([{"market": "ORD-BOS", "path_network": c,
                          "market_type": "nonstop_hub", "median_total_fare": 200.0,
                          "observed_offers": 100} for c in ["UA", "AA"]])
    config = {"core_carriers": ["UA", "AA"], "periods": 3, "assumed_conversion": 0.7,
              "fare_bound_multipliers": [0.25, 4], "assumed_cost_per_pax_mile": 0.04,
              "share_pseudocount": 1e-6, "Q2_calendar_days": 91,
              "outside_proxy_capacity": 1000000,
              "segments": [{"id": "leisure", "alpha": 0.02, "mix": 0.7,
                            "connection_penalty": 0.5, "carrier_pref": {}},
                           {"id": "business", "alpha": 0.008, "mix": 0.3,
                            "connection_penalty": 0.2, "carrier_pref": {}}],
              "airport_timezones": {"ORD": "America/Chicago", "BOS": "America/New_York"}}
    return legs, products, traffic, fares, config


def test_reference_choice_separates_outside_airline_from_nonpurchase():
    args = inputs()
    c = compile_market(initialize_cohort(*args, "2022-06-15"))
    prices = np.array([p.reference_fare for p in c.spec.products])
    for seg in c.spec.segments:
        probabilities = choice_probabilities(systematic_utilities(
            c, 0, seg.classes[0], prices, np.ones(len(prices), dtype=bool)))
        by_carrier = {p.carrier: probabilities[j] for j, p in enumerate(c.spec.products)}
        assert by_carrier == pytest.approx({"UA": 0.35, "AA": 0.21, "OUTSIDE": 0.14})
        assert probabilities[-1] == pytest.approx(0.30)
    row = c.spec.metadata["outside_traffic"][0]
    assert row["outside_network_Q2_passengers"] == 200
    assert row["outside_share_is_nonpurchase"] is False
    assert c.spec.metadata["calibrated"] is False
    assert c.spec.segments[0].classes[0].alpha != c.spec.segments[1].classes[0].alpha


def test_dated_physical_identity_and_prior_mean_are_preserved():
    args = inputs()
    c = compile_market(initialize_cohort(*args, "2022-06-15"))
    assert {l.id for l in c.spec.legs if l.carrier != "OUTSIDE"} == set(args[0].id)
    assert sum(l.carrier == "OUTSIDE" for l in c.spec.legs) == 1
    for leg in c.spec.legs[:2]:
        assert np.dot(leg.capacity.support, leg.capacity.probabilities) == 10.5
        assert leg.dep_time.endswith("-05:00") and leg.arr_time.endswith("-04:00")
    assert c.spec.airports[0].timezone == "America/Chicago"
    with pytest.raises(ValueError, match="exactly"):
        initialize_cohort(*args, "2022-07-01")


def test_fee_proxy_recovers_affine_gap_and_enforces_bounds():
    x = np.array([100.0, 150, 300, 450, 500])
    y = 0.075 * x + 7.5
    fit = fit_fee_moments(len(x), x.sum(), x @ x, y.sum(), x @ y, y @ y)
    assert fit["rate"] == pytest.approx(0.075, abs=1e-12)
    assert fit["flat"] == pytest.approx(7.5, abs=1e-10)
    assert fee_proxy(1, fit) == 1
    assert fee_proxy(200, fit) == pytest.approx(22.5)
    with pytest.raises(ValueError):
        fee_proxy(-1, fit)


def test_proxy_accounting_preserves_native_all_in_cash_and_is_analyst_only():
    c = compile_market(initialize_cohort(*inputs(), "2022-06-15"))
    engine = MarketEngine(c)
    engine.reset(400002)
    # A deterministic booking isolates accounting from sampling noise.
    book(engine.world, 0, c.product_index["UA_p"], 200.0,
         SimpleNamespace(rank=0, segment=0, taste_class=0), c)
    fits = {"UA|1": {"rate": 0.1, "flat": 5}, "AA|1": {"rate": 0.1, "flat": 5}}
    gross_before = engine.world.carriers[0].cash
    result = net_proxy_metrics(engine.world, c, fits)["UA"]
    assert result["net_base_fare_proxy"] == 175.0
    assert result["cash_identity_error"] == 0
    assert result["native_objective_uses_this_proxy"] is False
    assert engine.world.carriers[0].cash == gross_before == 200.0


@pytest.mark.parametrize("layout,mode", [("dense", "reference"), ("market_local_v1", "counted_sparse")])
def test_actual_prepare_and_full_tape_cohort_on_local_tables(tmp_path, layout, mode):
    legs, products, traffic, fares, cfg = inputs()
    network, targets = tmp_path / "network", tmp_path / "targets"
    network.mkdir(); targets.mkdir()
    legs.to_parquet(network / "legs.parquet"); products.to_parquet(network / "products.parquet")
    traffic.to_csv(targets / "q2_passenger_targets.csv", index=False)
    fares.to_csv(targets / "q2_quote_targets_by_market.csv", index=False)
    rows = [{"quarter": 2, "split": "calibration", "model_path_eligible": True,
             "path_network": c, "isNonStop": True, "totalFare": x, "baseFare": 0.9*x-5}
            for c in ["UA", "AA"] for x in [100.0, 200.0, 300.0]]
    source = tmp_path / "q2.parquet"; pd.DataFrame(rows).to_parquet(source)
    cfg.update(dated_network=str(network), Q2_targets=str(targets), Q2_quotes=str(source))
    path = tmp_path / "initializer.json"; atomic_json(path, cfg)
    out = tmp_path / "prepared"; out.mkdir()
    result = prepare({"parameters": {"initializer_config": str(path),
        "initializer_sha256": file_hash(path), "date": "2022-06-15"}},
        SimpleNamespace(directory=out))
    assert result["physical_legs"] == 2 and result["synthetic_outside_resources"] == 1
    assert result["calibrated"] is False and result["Q3_targets_used"] is False
    measured = tmp_path / "measured"; measured.mkdir()
    result = measure({"parameters": {"prepared_attempt": str(out),
        "spec_sha256": file_hash(out / "spec.json"), "seed": 400003, "run_cohort": True,
        "tape_layout": layout, "engine_mode": mode}},
        SimpleNamespace(directory=measured))
    assert result["tape_read_only"] and result["native_full_cohort_seconds"] > 0
    assert result["nonpurchase_requests"] >= 0 and result["outside_network_bookings"] >= 0
    assert result["contribution_proxy"]["UA"]["cash_identity_error"] == pytest.approx(0)


def two_market_compiled():
    """Include alternatives irrelevant to each exogenous potential request's market."""
    legs, products, traffic, fares, cfg = inputs()
    reverse = legs.copy()
    reverse["id"] += "_return"
    reverse["origin"], reverse["dest"] = "BOS", "ORD"
    other = products.copy()
    other["id"] += "_return"
    other["market"] = "BOS-ORD"
    other["leg_ids"] = [[c + "_regional_return"] for c in other.carrier]
    t = traffic.copy(); t["market"] = "BOS-ORD"
    f = fares.copy(); f["market"] = "BOS-ORD"
    return compile_market(initialize_cohort(pd.concat([legs, reverse]),
        pd.concat([products, other]), pd.concat([traffic, t]), pd.concat([fares, f]),
        cfg, "2022-06-15"))


def test_local_tape_preserves_all_non_taste_draws_and_is_read_only():
    from airline_market.core.rng import StreamRegistry
    from airline_market.sim.tape import ExogenousTape
    from airline_market.sim.market_local_tape import generate_market_local

    c = two_market_compiled()
    dense = ExogenousTape.generate(c, 400005, StreamRegistry(400005, "calibration/local_tape_test"))
    local = generate_market_local(c, 400005, StreamRegistry(400005, "calibration/local_tape_test"))
    for key, value in vars(dense).items():
        if key == "gumbel":
            continue
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(local, key))
        else:
            assert value == getattr(local, key)
    assert sum(v.nbytes for v in local.gumbel.values) < dense.gumbel.nbytes
    with pytest.raises(ValueError):
        local.gumbel.values[0][0] = 0


def test_local_tape_engine_path_matches_equivalent_dense_tastes_and_restore():
    from dataclasses import replace
    from airline_market.sim.market_local_tape import generate_market_local

    c = two_market_compiled()
    local = generate_market_local(c, 400006)
    values = np.zeros(local.gumbel.shape)
    for t, marks in enumerate(local.arrival_marks):
        for row, mark in enumerate(marks):
            cols = [*c.market_products[mark.market], len(c.spec.products)]
            values[t, row, cols] = local.gumbel[t, row, cols]
    values.setflags(write=False)
    dense = replace(local, gumbel=values)
    a, b = MarketEngine(c), MarketEngine(c)
    a.reset(400006, tape=local); b.reset(400006, tape=dense)
    a.step(); b.step()
    snapshot = a.snapshot()
    while not a.terminal():
        a.step(); b.step()
    assert a.trace() == b.trace()
    metrics = a.metrics()
    a.restore(snapshot)
    while not a.terminal():
        a.step()
    assert a.metrics() == metrics == b.metrics()
