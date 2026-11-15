"""A.10/A.5/A.14/B0/V9: NEW market/stage inconsistency witnesses and v2 checks."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import native_information_context as old
from airline_market.sim.proof_information import ProofInformationMarketEngine
from airline_market.sim.received_information import (
    DisclosureField,
    received_information_journal,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("NATIVE_CONTEXT_V2_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_native_context_v2", os.environ["NATIVE_CONTEXT_V2_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import native_information_context_v2 as module


def source(
    *, interline=True, connection=False, count=2, outside=False, shock=True, steps=3
):
    # NEW constructor/clock/namespace/primitive arrays, no prior fixture helper.
    spec = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        spec.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets},
    )
    c = compile_market(replace(spec, demand=demand))
    start = datetime(2026, 12, 15, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=h) for h in (0, 12, 24, 36)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 108))
        for leg in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    if shock:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
            ),
        )
    e = ProofInformationMarketEngine(
        scenario, dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=interline)
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(82974, "unit/V9_native_information_context_v2")
    )
    target = c.product_index["B_OD0_0" if connection else "B_OH0"]
    market = int(c.product_market[target])
    marks = tuple(
        ArrivalMark((i + 1) / (count + 1), market, 0, 0, 0) for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    if count:
        g[:, :, target] = 1e9
        if outside:
            g[:, :, -1] = 2e9
        elif count > 1:
            g[:, count - 1, -1] = 2e9
    g.setflags(write=False)
    e.reset(
        tape=replace(
            tape, arrival_marks=tuple(marks for _ in tape.arrival_marks), gumbel=g
        )
    )
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    for t in range(steps):
        if t == 1:
            for passenger in e._passengers.values():
                e.present_cancelled_itinerary(passenger.original, airport_counter=True)
        e.step(actions)
    j = received_information_journal(
        c,
        e.information_periods(),
        clock,
        [DisclosureField("notice", "text", "explicit future declared notice", "text")],
        definition="NEW native source",
        unobserved_definition="no actual knowledge/type/law supplied",
        row_limit=1000000,
    )
    return e, (
        j,
        e.own_information_periods(),
        e.proof_information_periods(),
        e.snapshot().native.histories,
        e.presentation_calls(),
    )


def join(inputs, limit=5000000):
    return module.native_information_contexts(
        *inputs, definition="NEW context consistency reader", row_limit=limit
    )


def inconsistent(inputs, kind):
    # NEW declared tampering of fresh unit receipts, never a native-produced bug.
    j, own, proof, histories, calls = inputs
    own, histories = list(own), list(histories)
    if kind == "ordinary_unoffered":
        context = old._encode(old.jsonable(j.analyst_context))
        row = context["periods"][0]["rows"][0]
        selected = context["native_spec"]["products"][
            row["analyst_person"]["native_row"]["chosen"]
        ]["id"]
        row["menu_view"]["offers"] = tuple(
            {**o, "offered": False} if o["product_id"] == selected else o
            for o in row["menu_view"]["offers"]
        )
        j = replace(j, analyst_context=context)
    else:
        root, facts, tickets, key = histories[0]
        index = next(
            i for i, f in enumerate(facts) if f.kind == "native_reentry_purchase"
        )
        products = j.analyst_context["native_spec"]["products"]
        original = products[
            next(i for i, p in enumerate(products) if p["id"] == facts[0].product_id)
        ]
        f = facts[index]
        if kind == "other_market":
            target = next(
                p["id"]
                for p in products
                if p["market"] != original["market"]
                and p["carrier"] != original["carrier"]
            )
            f = replace(f, product_id=target)
        elif kind == "own_label_other_carrier":
            f = replace(f, stage="own")
        elif kind == "interline_label_same_carrier":
            f = replace(f, product_id=original["id"])
        elif kind == "interline_disabled":
            period = facts[index].native_period
            row = own[period].reentry_rows[0]
            own[period] = replace(
                own[period],
                reentry_rows=(
                    replace(row, view=replace(row.view, interline_enabled=False)),
                ),
            )
        else:
            raise AssertionError(kind)
        histories[0] = root, (*facts[:index], f, *facts[index + 1 :]), tickets, key
    return j, own, proof, histories, calls


@pytest.mark.parametrize(
    "kind",
    [
        "other_market",
        "own_label_other_carrier",
        "interline_label_same_carrier",
        "interline_disabled",
        "ordinary_unoffered",
    ],
)
def test_new_inconsistent_declared_inputs_v1_accepts_v2_refuses(kind):
    _e, fresh = source()
    bad = inconsistent(fresh, kind)
    # ONE old pure consumer call on NEW inconsistent inputs is the bug witness.
    # No old source bank/result/acceptance/helper/analysis/review executes.
    accepted = old.native_information_contexts(
        *bad, definition="NEW mismatch witness", row_limit=5000000
    )
    assert len(accepted.people) == 6
    saved = pickle.dumps(bad)
    with pytest.raises(ValueError):
        join(bad)
    assert pickle.dumps(bad) == saved


@pytest.mark.parametrize("connection", [False, True])
def test_new_genuine_interline_replacement_market_carrier_fare_and_full_parent_identity(
    connection, monkeypatch
):
    e, inputs = source(connection=connection)
    saved = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())

    def fail(*args, **kwargs):
        raise AssertionError("pure v2 reader invoked source producer")

    monkeypatch.setattr(e, "step", fail)
    result = join(inputs)
    old_result = old.native_information_contexts(
        *inputs, definition="NEW context consistency reader", row_limit=5000000
    )
    assert old.native_information_context_record(
        result.native
    ) == old.native_information_context_record(old_result)
    assert len(result.purchase_support) == 6
    purchases = result.purchase_support[0]["ticket_purchases"]
    assert len(purchases) == 2 and purchases[-1]["stage"] == "interline"
    assert purchases[-1]["product_carrier"] != purchases[-1]["original_carrier"]
    assert all(p["request_market"] == p["product_market"] for p in purchases)
    assert len(result.native.periods) == 3
    assert (
        pickle.dumps(e.snapshot()) == saved
        and pickle.dumps(np.random.get_state()) == rng
    )
    json.dumps(module.native_information_context_record(result), allow_nan=False)
    assert (
        module.native_information_context_record(result)[
            "lifecycle_utility_CS_service_or_payoff"
        ]
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outside": True},
        {"count": 0},
        {"shock": False},
        {"interline": False},
        {"steps": 0},
        {"steps": 1},
    ],
)
def test_new_all_original_outside_empty_unstranded_failed_search_and_partial_support(
    kwargs,
):
    _e, inputs = source(**kwargs)
    result = join(inputs)
    assert len(result.purchase_support) == len(inputs[0].person_keys)
    assert len(result.native.periods) == len(inputs[1])
    for person in result.purchase_support:
        if person["analyst_root"] is None:
            assert person["ordinary_selected_product"] is None
            assert person["ordinary_selected_offer_enabled"] is None
            assert person["ticket_purchases"] == ()
    if kwargs.get("interline") is False:
        assert sum(len(p.reentry_contexts) for p in result.native.periods) == 2
        assert all(
            t["stage"] is None
            for p in result.purchase_support
            for t in p["ticket_purchases"]
        )


def test_new_parent_and_added_output_metadata_all_count_and_failure_owned(monkeypatch):
    _e, inputs = source()
    result = join(inputs)
    units = module._units(module.native_information_context_record(result))
    join(inputs, limit=units)
    with pytest.raises(old.NativeInformationContextLimit):
        join(inputs, limit=units - 1)
    parent_units = module._units(module.jsonable(result.native))
    assert parent_units < units
    with pytest.raises(old.NativeInformationContextLimit):
        join(inputs, limit=parent_units)
    out = module.native_information_context_record(result)
    out["result"]["purchase_support"][0]["ticket_purchases"][0]["product_market"] = (
        "mutated export"
    )
    assert (
        module.native_information_context_record(result)["result"]["purchase_support"][
            0
        ]["ticket_purchases"][0]["product_market"]
        != "mutated export"
    )
    saved = pickle.dumps(inputs)

    def fail(*args):
        raise RuntimeError("NEW injected market/stage validation failure")

    monkeypatch.setattr(module, "_purchase_support", fail)
    with pytest.raises(RuntimeError, match="validation failure"):
        join(inputs)
    assert pickle.dumps(inputs) == saved


def test_new_v2_executes_join_once_on_new_input(monkeypatch):
    _e, inputs = source()
    calls = []
    original = module._join_current_prefix

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_join_current_prefix", record)
    result = join(inputs)
    assert calls == [1] and result.rule == module.RULE
    with pytest.raises(TypeError):
        result.purchase_support[0]["analyst_root"] = 99
    with pytest.raises(TypeError):
        module.native_information_context_record(result.native)


def test_new_genuine_closing_boundary_cancellation_kept_future_search_not_added():
    _e, inputs = source(steps=1)
    n = len(inputs[1])
    facts = inputs[3][0][1]
    assert facts[-1].kind == "native_ticket_cancelled" and facts[-1].native_period == n
    with pytest.raises(ValueError, match="beyond complete closed prefix"):
        old.native_information_contexts(
            *inputs, definition="NEW actual boundary witness", row_limit=5000000
        )
    result = join(inputs)
    retained = result.native.analyst_context["final_prospective_histories"][0][1][-1]
    assert (
        retained["native_period"] == n and retained["kind"] == "native_ticket_cancelled"
    )
    assert (
        len(result.native.periods) == n
        and not result.native.periods[0].reentry_contexts
    )
    root, source_facts, tickets, key = inputs[3][0]
    future = replace(
        source_facts[-1],
        ordinal=len(source_facts),
        kind="native_reentry_outside",
        person_ticket=None,
        product_id=None,
        fare=None,
        stage=None,
    )
    bad = (*inputs[:3], ((root, (*source_facts, future), tickets, key),), inputs[-1])
    with pytest.raises(ValueError, match="only cancellation"):
        join(bad)
