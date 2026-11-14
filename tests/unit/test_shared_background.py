"""Resource coupling, information timing and replay for excluded ticket load."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from airline_market.campaign.instances import build_instance
from airline_market.pipeline.background_load import generate_load_events
from airline_market.sim.shared_background import (
    LoadEvent, SharedLoadEngine, augment_load_products, PREFIX,
)
from airline_market.sim.engine import MarketEngine
from airline_market.sim.market_local_tape import generate_market_local
from airline_market.sim.ledger import inventory, resolve
from airline_market.spec import compile_market, dump_spec
from airline_market.spec.model import PeriodSchedule, CapacityEventSpec, InformationArmSpec
from airline_market.core.hashing import canonical_json
from airline_market.campaign.artifacts import file_hash


def fixture_spec():
    """Use legal Poisson inputs, no unknown penalty allocation, and known seats."""
    s = build_instance("hub_feeder_family", {
        "T": 3, "n_spokes": 2, "capacity": 2, "capacity_support": [2]})
    d = replace(s.demand, mode="poisson", arrival_probability=None, market_weights=None,
                intensity={m.id: PeriodSchedule.parse({"constant": 0.4}) for m in s.markets})
    assignments = tuple(replace(a, controller="fixed", params={"theta": [0.0] *
        len(next(p for p in s.pricing if p.carrier == a.carrier).theta_bounds)})
        for a in s.controllers)
    return replace(s, demand=d, controllers=assignments,
                   resolution=replace(s.resolution, penalty_per_cancellation=0))


def compiled():
    """Preserve the original complete strategic network and append ledger products."""
    return compile_market(augment_load_products(fixture_spec(), ["A", "B"]))


def test_augmentation_preserves_original_indices_and_no_customer_load_demand():
    s = fixture_spec()
    before = dump_spec(s)
    a = augment_load_products(s, ["A", "B"])
    assert dump_spec(s) == before
    assert a.products[:len(s.products)] == s.products
    c = compile_market(a)
    assert np.array_equal(c.A_global[:, :len(s.products)], compile_market(s).A_global)
    for p in a.products[len(s.products):]:
        assert not p.controlled and p.service_cost == 0
        assert a.demand.intensity[p.market].value_at(0) == 0
        assert not next(m for m in a.markets if m.id == p.market).in_segment
    assert a.quotes == s.quotes


def test_joint_admission_is_atomic_across_carriers_and_repeated_resource_use():
    c = compiled()
    events = (LoadEvent(0, 0, ("A_OH0", "B_HD0")),
              LoadEvent(1, 0, ("A_OH0", "A_OH0")))
    e = SharedLoadEngine(c, events)
    e.reset(200020, tape=generate_market_local(c, 200020))
    e.begin_period()
    assert e.load_counts["accepted_groups"] == 1
    assert e.load_counts["capacity_rejected_groups"] == 1
    assert len(e.world.carriers[0].bookings) == len(e.world.carriers[1].bookings) == 1
    assert inventory(e.world.carriers[0], c, 0)[0] == 1
    assert sum(s.cash for s in e.world.carriers) == 0
    assert e.metrics()["journeys"] == 0
    assert e.metrics()["profit"] == 0


def test_canceled_component_releases_whole_cross_network_group_without_fake_economics():
    c = compiled()
    e = SharedLoadEngine(c, (LoadEvent(0, 0, ("A_OH0", "B_HD0")),))
    e.reset(200021, tape=generate_market_local(c, 200021))
    e.begin_period()
    e.world.carriers[0].capacity[0] = 0
    resolve(e.world, 0, c)
    e.reconcile_load_groups()
    assert all(not s.bookings for s in e.world.carriers)
    assert e.canceled_groups == {0}
    result = e.metrics()
    assert result["cancellations"] == 0
    assert result["shared_core_load"]["by_carrier"]["B"]["canceled_components"] == 1
    assert result["shared_core_load"]["by_carrier"]["A"]["attributed_revenue"] is None
    assert result["profit"] == 0
    e.reconcile_load_groups()
    assert e.world.carriers[1].canceled == 1


def test_future_load_is_not_applied_early_and_snapshot_branch_do_not_duplicate_it():
    c = compiled()
    events = (LoadEvent(0, 1, ("A_OH0",)),)
    e = SharedLoadEngine(c, events)
    e.reset(200022, tape=generate_market_local(c, 200022))
    e.begin_period()
    assert not e.world.carriers[0].bookings
    e.step(_requests=[])
    saved = e.snapshot()
    branch = e.branch()
    e.begin_period(); e.begin_period()
    assert len(e.world.carriers[0].bookings) == 1
    branch.begin_period()
    assert branch.metrics() == e.metrics()
    e.restore(saved)
    assert not e.world.carriers[0].bookings
    e.begin_period()
    assert len(e.world.carriers[0].bookings) == 1
    assert e.metrics() == branch.metrics()


def test_zero_load_reproduces_original_strategic_path_on_named_market_local_draws():
    original = compile_market(fixture_spec())
    augmented = compiled()
    a = MarketEngine(original)
    b = SharedLoadEngine(augmented, (LoadEvent(0, 0, ("A_OH0",)),), fraction=0)
    a.reset(200023, tape=generate_market_local(original, 200023))
    b.reset(200023, tape=generate_market_local(augmented, 200023))
    while not a.terminal():
        a.step(); b.step()
        assert a.world.request_log == b.world.request_log
        for i in range(2):
            assert a.world.carriers[i].bookings == b.world.carriers[i].bookings
            np.testing.assert_array_equal(a.world.carriers[i].posted_prices,
                b.world.carriers[i].posted_prices[:len(original.products_of[i])])
    assert a.metrics()["carriers"] == b.metrics()["carriers"]


def test_pattern_generator_is_deterministic_and_preserves_unmatched_mass():
    c = compiled()
    patterns = [
        {"category": "represented_core", "expanded_passengers": 100000,
         "ordered_core_coupon_edges": [["A", "O0", "H"]]},
        {"category": "off_pool_od", "expanded_passengers": 9100,
         "ordered_core_coupon_edges": [["A", "O0", "H"], ["B", "H", "D0"]]},
        {"category": "off_pool_od", "expanded_passengers": 910,
         "ordered_core_coupon_edges": [["A", "MISSING", "H"]]}]
    events, report = generate_load_events(c, patterns, 400002)
    assert (events, report) == generate_load_events(c, patterns, 400002)
    assert report["excluded_Q2_ticket_pattern_mass"] == 10010
    assert report["covered_Q2_ticket_pattern_mass"] == 9100
    assert report["unmatched_patterns"][0]["expanded_ticket_passengers"] == 910
    assert not report["remaining_T100_residual_used"]
    assert all(e.legs == ("A_OH0", "B_HD0") for e in events)
    assert len(events) > 0 and set(e.period for e in events) == {0, 1, 2}


def test_unidentified_penalty_allocation_and_invalid_load_inputs_are_explicit():
    s = fixture_spec()
    with pytest.raises(ValueError, match="penalties"):
        augment_load_products(replace(s, resolution=replace(s.resolution,
            penalty_per_cancellation=1)), ["A"])
    c = compiled()
    with pytest.raises(ValueError, match="invalid or duplicate"):
        SharedLoadEngine(c, (LoadEvent(0, 3, ("A_OH0",)),))
    with pytest.raises(ValueError, match="fraction"):
        SharedLoadEngine(c, (), fraction=1.1)
    with pytest.raises(ValueError, match="weights"):
        generate_load_events(c, [], 400002, period_weights=[-1, 1, 1])


def test_native_capacity_event_cancels_entire_group_and_excludes_unidentified_cost():
    s = fixture_spec()
    s = replace(s, legs=tuple(replace(l, cost_per_pax=0.37) for l in s.legs),
        capacity_process=replace(s.capacity_process, events=(CapacityEventSpec(
            trigger={"hazard": {"probability": 1.0}}, effects=(("A_OH0", -2),)),)))
    c = compile_market(augment_load_products(s, ["A", "B"]))
    e = SharedLoadEngine(c, (LoadEvent(0, 0, ("A_OH0", "B_HD0")),))
    e.reset(200024, tape=generate_market_local(c, 200024))
    e.begin_period()
    assert e.metrics()["profit"] == 0
    e.step(_requests=[])
    assert all(not carrier.bookings for carrier in e.world.carriers)
    assert e.load_counts["canceled_groups"] == 1
    assert e.metrics()["profit"] == e.metrics()["cancellations"] == 0


def test_private_rival_load_and_future_events_do_not_enter_own_only_observation():
    c = compiled()
    empty = SharedLoadEngine(c)
    loaded = SharedLoadEngine(c, (LoadEvent(0, 0, ("B_HD0",)),
                                  LoadEvent(1, 2, ("A_OH0",))))
    for e in (empty, loaded):
        e.reset(200025, tape=generate_market_local(c, 200025))
    arm = InformationArmSpec(name="own_only", observer="A", requests=())
    a, b = empty.observe("A", arm), loaded.observe("A", arm)
    assert canonical_json(a) == canonical_json(b)
    assert not loaded.world.carriers[0].bookings
    assert len(loaded.world.carriers[1].bookings) == 1


def test_actual_bounded_study_keeps_tapes_matched_and_resource_totals_separate(tmp_path):
    from airline_market.experiments.shared_load_study import measure

    spec = tmp_path / "original.json"
    spec.write_text(dump_spec(fixture_spec()))
    patterns = tmp_path / "patterns.json"
    patterns.write_text(json.dumps([{"category": "off_pool_od",
        "expanded_passengers": 9100,
        "ordered_core_coupon_edges": [["A", "O0", "H"], ["A", "H", "D0"]]}]))
    p = {"source_spec": str(spec), "source_spec_sha256": file_hash(spec),
         "source_patterns": str(patterns), "source_patterns_sha256": file_hash(patterns),
         "seed": 400002, "quarter_days": 91, "core_carriers": ["A"],
         "outside_carrier": "B"}
    results = []
    for fraction in [0.0, 1.0]:
        directory = tmp_path / str(fraction)
        directory.mkdir()
        results.append(measure({"parameters": {**p, "fraction": fraction}},
                               SimpleNamespace(directory=directory)))
    a, b = results
    assert a["spec_hash"] == b["spec_hash"]
    assert a["mapping"]["event_tape_sha256"] == b["mapping"]["event_tape_sha256"]
    assert a["potential_strategic_requests"] == b["potential_strategic_requests"]
    assert a["resource_accounting_exact"] and b["resource_accounting_exact"]
    assert b["shared_core_load"]["accepted_components"] > 0
    assert b["shared_core_load"]["capacity_rejected_groups"] > 0
    assert not b["calibrated"] and not b["remaining_residual_imputed"]


def test_external_load_records_sellout_even_without_any_strategic_requests():
    c = compiled()
    e = SharedLoadEngine(c, (LoadEvent(0, 0, ("A_OH0", "A_OH0")),))
    e.reset(200026, tape=generate_market_local(c, 200026))
    e.step(_requests=[])
    assert inventory(e.world.carriers[0], c, 0)[0] == 0
    assert e.world.sellout_period[c.leg_index["A_OH0"]] == 0
