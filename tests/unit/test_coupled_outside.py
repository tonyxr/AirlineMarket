"""Single-passenger resource coupling, refunds, replay and conditional source laws."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from airline_market.campaign.instances import build_instance
from airline_market.campaign.artifacts import file_hash
from airline_market.core.hashing import canonical_json
from airline_market.pipeline.outside_footprints import build_catalogue, generate_purchase_tape
from airline_market.sim import engine as native
from airline_market.sim.arrivals import Request
from airline_market.sim.coupled_outside import CoupledOutsideEngine, PurchaseFootprint
from airline_market.sim.shared_background import SharedLoadEngine, augment_load_products
from airline_market.sim.tape import ExogenousTape
from airline_market.sim.ledger import inventory
from airline_market.spec import compile_market, dump_spec, load_spec
from airline_market.spec.model import InformationArmSpec, PeriodSchedule


def fixture_spec(event=None):
    """A small fixed-policy network with an explicit third-carrier outside ticket."""
    s = build_instance("hub_feeder_family", {
        "T": 3, "n_spokes": 2, "capacity": 2, "capacity_support": [2]})
    d = replace(s.demand, mode="poisson", arrival_probability=None, market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 0.4}) for m in s.markets})
    data = json.loads(dump_spec(replace(s, demand=d)))
    data["resolution"]["penalty_per_cancellation"] = 0
    data["capacity_process"]["events"] = [event] if event else []
    for assignment in data["controllers"]:
        assignment["controller"] = "fixed"
        pricing = next(p for p in data["pricing"] if p["carrier"] == assignment["carrier"])
        assignment["params"] = {"theta": [0.0] * len(pricing["theta_bounds"])}
    data["carriers"].append({"code": "OUTSIDE", "kind": "outside_network_proxy"})
    base = {}
    for k, market in enumerate(s.markets):
        identifier = "OUTSIDE_" + market.id
        data["legs"].append({"id": identifier, "carrier": "OUTSIDE",
            "origin": market.origin, "dest": market.dest,
            "capacity": {"support": [20], "probabilities": [1.0], "public": True}})
        data["products"].append({"id": identifier, "carrier": "OUTSIDE", "market": market.id,
            "legs": [identifier], "service_cost": 0, "fare_floor": 100,
            "fare_cap": 100, "reference_fare": 100, "priority": 1000 + k,
            "controlled": False})
        base[identifier] = 100
    data["pricing"].append({"carrier": "OUTSIDE", "base": base,
        "controllable_features": [], "theta_bounds": [], "theta_grid": [[]]})
    data["controllers"].append({"carrier": "OUTSIDE", "controller": "fixed", "params": {"theta": []}})
    data["objectives"].append({"carrier": "OUTSIDE", "profit_scale": 1000})
    data["metadata"]["outside_traffic"] = [{"market": "OD0_0", "outside_network_Q2_passengers": 50.0}]
    return load_spec(data)


def setup(mark=PurchaseFootprint(("A_OH0", "B_HD0")), event=None):
    """Use one explicitly marked request with an unchanged dense native taste tape."""
    c = compile_market(augment_load_products(fixture_spec(event), ["A", "B"]))
    marks = {(0, 0): mark} if mark is not None else {}
    e = CoupledOutsideEngine(c, footprint_tape=marks)
    e.reset(200030, tape=ExogenousTape.generate(c, 200030))
    m = c.market_index["OD0_0"]
    return c, e, Request(0, m, 0, 0, 0, 0), c.product_index["OUTSIDE_OD0_0"]


def test_one_choice_one_ticket_and_atomic_cross_carrier_resource_attachment():
    c, e, request, outside = setup()
    result = e.step(_requests=[request], _choices=[outside])
    assert len(result.purchases) == len(e.world.request_log) == 1
    assert [len(s.bookings) for s in e.world.carriers] == [1, 1, 1]
    assert e.world.carriers[2].cash == 100
    assert e.world.carriers[0].cash == e.world.carriers[1].cash == 0
    assert e.metrics()["requests"] == 1
    assert e.metrics()["carriers"]["OUTSIDE"]["journeys"] == 1
    assert e.metrics()["carriers"]["A"]["journeys"] == 0
    assert e.metrics()["outside_resource_coupling"]["coupled_core_components"] == 2


def test_capacity_censoring_occurs_before_native_choice_and_welfare():
    c, e, request, outside = setup()
    e.world.carriers[0].capacity[0] = 0
    e.step(_requests=[request])
    assert e.world.request_log[0]["chosen"] != outside
    assert e.purchase_counts["capacity_blocked_opportunities"] == 1
    assert not e.purchase_links


def test_forced_infeasible_choice_has_no_partial_receipt_and_restores_aliases():
    c, e, request, outside = setup()
    e.world.carriers[0].capacity[0] = 0
    old_book, old_events = native.book, native.apply_events
    with pytest.raises(ValueError, match="complete resource footprint"):
        e.step(_requests=[request], _choices=[outside])
    assert all(not s.bookings and s.cash == 0 for s in e.world.carriers)
    assert native.book is old_book and native.apply_events is old_events
    assert e._pending is None


@pytest.mark.parametrize("affected", ["A_OH0", "OUTSIDE_OD0_0"])
def test_native_disruption_cancels_whole_ticket_refunds_once_before_callbacks(affected):
    event = {"trigger": {"hazard": {"probability": 1.0}},
             "effects": [[affected, -20 if affected.startswith("OUTSIDE") else -2]]}
    c, e, request, outside = setup(event=event)
    result = e.step(_requests=[request], _choices=[outside])
    assert all(not s.bookings for s in e.world.carriers)
    ticket = e.world.carriers[2]
    assert ticket.revenue == ticket.refunds == 100 and ticket.cash == 0
    assert result.cash_changes == (0.0, 0.0, 0.0)
    e.reconcile_load_groups()
    assert ticket.refunds == 100
    assert e.metrics()["carriers"]["OUTSIDE"]["cancellations"] == 1
    assert e.metrics()["carriers"]["A"]["cancellations"] == 0
    assert e.purchase_counts["canceled_coupled_tickets"] == 1


def test_snapshot_branch_and_restore_preserve_ticket_component_links():
    c, e, request, outside = setup()
    e.step(_requests=[request], _choices=[outside])
    snapshot = e.snapshot()
    branch = e.branch()
    e.step(_requests=[])
    branch.step(_requests=[])
    assert e.metrics() == branch.metrics()
    e.restore(snapshot)
    e.step(_requests=[])
    assert e.metrics() == branch.metrics()
    assert len(e.purchase_links) == 1


def test_empty_coupling_matches_native_shared_load_requests_and_economics():
    c, e, _, _ = setup(mark=None)
    baseline = SharedLoadEngine(c)
    baseline.reset(200030, tape=ExogenousTape.generate(c, 200030))
    while not e.terminal():
        e.step()
        baseline.step()
        assert e.world.request_log == baseline.world.request_log
        assert [s.bookings for s in e.world.carriers] == [s.bookings for s in baseline.world.carriers]
    result = e.metrics()
    result.pop("outside_resource_coupling")
    assert result == baseline.metrics()


def test_future_or_private_purchase_marks_do_not_enter_own_only_observation():
    c, e, _, _ = setup()
    other = CoupledOutsideEngine(c, footprint_tape={(2, 0): PurchaseFootprint(("B_HD1",))})
    other.reset(200030, tape=ExogenousTape.generate(c, 200030))
    arm = InformationArmSpec(name="own", observer="A", requests=())
    assert canonical_json(e.observe("A", arm)) == canonical_json(other.observe("A", arm))
    assert not e.world.carriers[0].bookings


def source_frame():
    """Two represented carriers and an outside pool containing one core footprint."""
    common = {"year": 2022, "quarter": 2, "split": "calibration", "market": "OD0_0",
              "sequence_valid": True}
    return pd.DataFrame([{**common, **r} for r in [
        {"passengers_expanded": 60, "model_path_eligible": True, "path_network": "A",
         "airport_sequence": "O0:H:D0", "ticket_carrier_sequence": "A:A"},
        {"passengers_expanded": 40, "model_path_eligible": True, "path_network": "B",
         "airport_sequence": "O0:H:D0", "ticket_carrier_sequence": "B:B"},
        {"passengers_expanded": 20, "model_path_eligible": False, "path_network": "MIXED",
         "airport_sequence": "O0:H:D0", "ticket_carrier_sequence": "A:X"},
        {"passengers_expanded": 30, "model_path_eligible": False, "path_network": "X",
         "airport_sequence": "O0:H:D0", "ticket_carrier_sequence": "X:X"}]])


def test_q2_catalogue_conditions_on_same_outside_population_and_filters_other_carriers():
    s = fixture_spec()
    routes = {("A", "O0", "H"), ("X", "O0", "H"), ("X", "H", "D0")}
    result = build_catalogue(source_frame(), s, routes, ["A", "B"])
    market = result["markets"]["OD0_0"]
    assert market["outside_Q2_ticket_passengers"] == 50
    assert market["identified_core_path_probability"] == 0.4
    assert result["core_coupon_edges_by_carrier"] == {"A": 20}
    assert not result["arrival_population_added"]
    bad = source_frame()
    bad.loc[3, "passengers_expanded"] = 31
    with pytest.raises(ValueError, match="denominator changed"):
        build_catalogue(bad, s, routes, ["A", "B"])
    with pytest.raises(ValueError, match="Q2"):
        build_catalogue(source_frame().assign(quarter=3), s, routes, ["A", "B"])


def test_latent_subtype_law_is_named_and_unmapped_paths_are_infeasible():
    c, e, _, _ = setup()
    m = c.market_index["OD0_0"]
    fake = SimpleNamespace(arrival_marks=(tuple(SimpleNamespace(market=m) for _ in range(10000)),))
    catalogue = {"markets": {"OD0_0": {"paths": [{"core_coupon_edges": [["A", "O0", "H"]],
        "probability_given_outside": 0.4}]}}}
    marks, info = generate_purchase_tape(c, fake, catalogue, 400003)
    assert (marks, info) == generate_purchase_tape(c, fake, catalogue, 400003)
    assert abs(len(marks) - 4000) <= 4 * np.sqrt(10000 * 0.4 * 0.6)
    assert all(mark.legs == ("A_OH0",) and mark.mapped for mark in marks.values())
    catalogue["markets"]["OD0_0"]["paths"][0]["core_coupon_edges"] = [["A", "MISSING", "H"]]
    marks, info = generate_purchase_tape(c, fake, catalogue, 400003)
    assert marks and all(not mark.mapped for mark in marks.values())
    assert info["unmapped_potential_footprints"] == len(marks)


def test_actual_probe_reuses_physical_baseline_and_conserves_passengers(tmp_path):
    from airline_market.experiments.shared_load_study import measure as baseline_measure
    from airline_market.experiments.outside_coupling_study import measure

    spec = tmp_path / "original.json"
    spec.write_text(dump_spec(fixture_spec()))
    patterns = tmp_path / "external.json"
    patterns.write_text("[]\n")
    markets = tmp_path / "markets.parquet"
    source_frame().to_parquet(markets)
    capacity = tmp_path / "capacity.csv"
    pd.DataFrame([{"marketing_carrier": leg.carrier, "origin": leg.origin, "dest": leg.dest}
                  for leg in fixture_spec().legs if leg.carrier in ["A", "B"]]).to_csv(capacity, index=False)
    common = {"source_spec": str(spec), "source_spec_sha256": file_hash(spec),
        "source_patterns": str(patterns), "source_patterns_sha256": file_hash(patterns),
        "core_carriers": ["A", "B"], "outside_carrier": "OUTSIDE",
        "quarter_days": 91, "seed": 400002, "fraction": 1.0}
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = baseline_measure({"parameters": common}, SimpleNamespace(directory=baseline_dir))
    baseline_path = baseline_dir / "sealed_fixture_result.json"
    baseline_path.write_text(json.dumps(baseline))
    directory = tmp_path / "coupled"
    directory.mkdir()
    p = {**common, "Q2_scoped_markets": str(markets), "Q2_scoped_markets_sha256": file_hash(markets),
        "Q2_capacity_targets": str(capacity), "Q2_capacity_targets_sha256": file_hash(capacity),
        "baseline_result": str(baseline_path), "baseline_result_sha256": file_hash(baseline_path)}
    result = measure({"parameters": p}, SimpleNamespace(directory=directory))
    assert result["one_choice_one_ticket"] and result["core_resource_accounting_exact"]
    assert result["potential_requests"] == baseline["potential_strategic_requests"]
    assert result["conditional_core_coupon_edges"] == {"A": 20}
    assert not result["new_passenger_population"] and not result["calibrated"]
