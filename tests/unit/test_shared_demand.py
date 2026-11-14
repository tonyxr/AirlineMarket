"""A.5/B1/B0/V9: declared shared Poisson envelope and native projection."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.sim.arrivals import period_requests
from airline_market.sim.exogenous import demand_multiplier
from airline_market.sim.tape import ExogenousTape, segment_weights
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule
from tests.unit import test_potential_lineage as lineage
from tests.unit import test_rationing_native as fixture_lib

if os.environ.get("SHARED_DEMAND_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_shared_demand", os.environ["SHARED_DEMAND_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import shared_demand as module


def members():
    base = fixture_lib.fixture().compiled.spec
    normal = base.exogenous.states[0]
    exo = replace(
        base.exogenous,
        states=(replace(normal, name="low"), replace(normal, name="high")),
        initial=(1.0, 0.0),
    )
    base = replace(base, exogenous=exo)
    a = replace(
        base.demand,
        intensity={m.id: PeriodSchedule.parse({"constant": 3.0}) for m in base.markets},
        segment_weights=PeriodSchedule.parse({"constant": (0.7, 0.3)}),
        exogenous_multipliers={"low": 0.5, "high": 1.0},
    )
    b = replace(
        a,
        intensity={m.id: PeriodSchedule.parse({"constant": 6.0}) for m in base.markets},
        segment_weights=PeriodSchedule.parse({"constant": (0.4, 0.6)}),
        holiday_multiplier=1.3,
    )
    return {
        "baseline": compile_market(replace(base, demand=a)),
        "surge": compile_market(replace(base, demand=b)),
    }


def bank(inputs=None):
    return module.shared_poisson_bank(
        members() if inputs is None else inputs,
        seed=31,
        namespace="unit/V9_shared_demand",
    )


def test_every_declared_native_rate_is_dominated_without_metric_or_count_changes():
    inputs = members()
    b = bank(inputs)
    for name, c in inputs.items():
        cap = dict(b.member_caps)[name]
        for t in range(5):
            for m, market in enumerate(c.spec.markets):
                for g, w in enumerate(segment_weights(c, t, m)):
                    expected = (
                        c.spec.demand.intensity[market.id].value_at(t)
                        * max(demand_multiplier(c, e) for e in range(2))
                        * w
                    )
                    assert (
                        cap[t, m, g] == expected
                        and b.envelope_rates[t, m, g] >= expected
                    )
                    for e in range(2):
                        target = (
                            c.spec.demand.intensity[market.id].value_at(t)
                            * w
                            * demand_multiplier(c, e)
                        )
                        reconstructed = (
                            b.envelope_rates[t, m, g]
                            * (cap[t, m, g] / b.envelope_rates[t, m, g])
                            * demand_multiplier(c, e)
                            / max(demand_multiplier(c, i) for i in range(2))
                        )
                        assert reconstructed == pytest.approx(
                            target, rel=1e-14, abs=1e-14
                        )


def test_actual_native_acceptance_matches_independent_shared_uniform_predicate():
    inputs = members()
    b = bank(inputs)
    for name, c in inputs.items():
        tape = b.project(name)
        caps = dict(b.member_caps)[name]
        for t in range(5):
            for e in range(2):
                factor = demand_multiplier(c, e) / max(
                    demand_multiplier(c, i) for i in range(2)
                )
                expected = [
                    slot
                    for slot, m in enumerate(b.tape.arrival_marks[t])
                    if m.thinning
                    < caps[t, m.market, m.segment]
                    / b.envelope_rates[t, m.market, m.segment]
                    * factor
                ]
                requests = period_requests(c, tape, t, e)
                assert [r.gumbel_row for r in requests] == expected
                assert [r.rank for r in requests] == list(range(len(expected)))


def test_projection_retains_every_potential_row_and_original_nonacceptance_mark():
    b = bank()
    a = b.project("baseline")
    c = b.project("surge")
    for original, x, y in zip(b.tape.arrival_marks, a.arrival_marks, c.arrival_marks):
        assert len(original) == len(x) == len(y)
        for o, p, q in zip(original, x, y):
            assert (
                (o.time, o.market, o.segment, o.class_uniform)
                == (p.time, p.market, p.segment, p.class_uniform)
                == (q.time, q.market, q.segment, q.class_uniform)
            )
            assert 0 <= p.thinning <= 1 and 0 <= q.thinning <= 1


def test_no_auxiliary_redraw_or_array_copy_on_projection(monkeypatch):
    b = bank()
    before = pickle.dumps(np.random.get_state())
    monkeypatch.setattr(
        ExogenousTape,
        "generate",
        classmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("redraw"))),
    )
    a = b.project("baseline")
    c = b.project("surge")
    for key, value in vars(b.tape).items():
        if isinstance(value, np.ndarray):
            assert (
                getattr(a, key) is value
                and getattr(c, key) is value
                and not value.flags.writeable
            )
    with pytest.raises(ValueError):
        a.gumbel[0, 0, 0] = 0
    assert pickle.dumps(np.random.get_state()) == before


def test_one_native_generation_and_scenario_registration_order_independence(
    monkeypatch,
):
    inputs = members()
    native = ExogenousTape.generate
    calls = []

    def generate(cls, *a, **k):
        calls.append(1)
        return native(*a, **k)

    monkeypatch.setattr(ExogenousTape, "generate", classmethod(generate))
    a = bank(inputs)
    assert calls == [1]
    b = bank(dict(reversed(list(inputs.items()))))
    assert a.record() == b.record() and pickle.dumps(a.tape) == pickle.dumps(b.tape)
    assert a.project("baseline").arrival_marks == b.project("baseline").arrival_marks


def test_identical_members_share_identical_projected_tapes():
    c = members()["baseline"]
    b = bank({"a": c, "b": c})
    a = b.project("a")
    d = b.project("b")
    assert pickle.dumps(a) == pickle.dumps(d)
    for t in range(5):
        assert period_requests(c, a, t, 0) == period_requests(c, d, t, 0)


def test_zero_intensity_strata_keep_slots_but_never_admit_fake_demand():
    inputs = members()
    c = inputs["baseline"]
    zero = replace(
        c.spec.demand,
        intensity={
            m.id: PeriodSchedule.parse({"constant": 0.0}) for m in c.spec.markets
        },
    )
    inputs["zero"] = compile_market(replace(c.spec, demand=zero))
    b = bank(inputs)
    tape = b.project("zero")
    assert sum(map(len, tape.arrival_marks)) > 0
    assert all(
        not period_requests(inputs["zero"], tape, t, e)
        for t in range(5)
        for e in range(2)
    )
    assert all(m.thinning == 1.0 for row in tape.arrival_marks for m in row)


def test_segment_specific_envelope_handles_disjoint_member_segment_support():
    inputs = members()
    c = inputs["baseline"]
    a = compile_market(
        replace(
            c.spec,
            demand=replace(
                c.spec.demand,
                segment_weights=PeriodSchedule.parse({"constant": (1.0, 0.0)}),
            ),
        )
    )
    z = compile_market(
        replace(
            c.spec,
            demand=replace(
                c.spec.demand,
                segment_weights=PeriodSchedule.parse({"constant": (0.0, 1.0)}),
            ),
        )
    )
    b = bank({"first": a, "second": z})
    assert all(
        r.segment == 0
        for t in range(5)
        for r in period_requests(a, b.project("first"), t, 1)
    )
    assert all(
        r.segment == 1
        for t in range(5)
        for r in period_requests(z, b.project("second"), t, 1)
    )


def test_non_demand_schema_change_and_unknown_member_are_not_filled():
    inputs = members()
    c = inputs["baseline"]
    bad = compile_market(
        replace(c.spec, choice=replace(c.spec.choice, connection_penalty_rule="none"))
    )
    with pytest.raises(ValueError, match="non-demand"):
        bank({"a": c, "b": bad})
    with pytest.raises(KeyError, match="absent"):
        bank().project("unregistered")


def test_empty_duplicate_or_bernoulli_family_and_unnamed_stream_refused():
    c = members()["baseline"]
    with pytest.raises(ValueError):
        bank({})
    with pytest.raises(ValueError, match="distinct"):
        bank([("same", c), ("same", c)])
    with pytest.raises(ValueError, match="namespace"):
        module.shared_poisson_bank({"a": c}, seed=31, namespace=" ")
    old, _, _ = lineage.dated.fixture(trigger=False)
    with pytest.raises(ValueError, match="Poisson"):
        bank({"notpoisson": old.compiled})


def test_all_zero_global_multiplier_refused_instead_of_native_divide_zero():
    c = members()["baseline"]
    d = replace(c.spec.demand, exogenous_multipliers={"low": 0.0, "high": 0.0})
    # Compile stays native; the new adapter reports its unsupported zero divisor.
    zero = compile_market(replace(c.spec, demand=d))
    with pytest.raises(ValueError, match="positive maximum"):
        bank({"zero": zero})


def test_profile_json_retains_scope_and_complete_declared_rates():
    b = bank()
    x = json.loads(json.dumps(b.record()))
    assert (
        set(x["member_spec_sha256"]) == {"baseline", "surge"}
        and len(x["envelope_rates"]) == 5
    )
    assert x["projection_rng_calls"] == 0 and x["generation_raw_rng_calls"] is None
    assert not x["old_tapes_rewritten"] and not x["primary_B1_or_empirical_adoption"]


def test_projected_member_tape_executes_native_bookings_with_original_slots():
    from airline_market.policy.potential_lineage import PotentialLineageMarketEngine

    inputs = members()
    b = bank(inputs)
    for name, c in inputs.items():
        e = PotentialLineageMarketEngine(c)
        e.reset(tape=b.project(name))
        e.world.exo = 0
        requests = tuple(period_requests(c, e.tape, 0, 0))
        expected = {(0, r.gumbel_row, r.market): r for r in requests}
        e.step(fixture_lib.actions(e))
        assert len(e.world.request_log) == len(requests)
        for record, coordinate in e.coordinate_records():
            assert coordinate.stream_key in expected
            assert coordinate.native_rank == expected[coordinate.stream_key].rank
        assert len(e.coordinate_records()) == e.world.next_record_id
