"""A.10/A.5/B0/V9/B4: NEW conditional finite source118974, no seed/tape."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim import source_future_transport as st
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliFutureResult,
    BernoulliOpportunity,
    bernoulli_future_expectation,
)
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 10**10
CELLS = (
    "cancel_closed",
    "cancel_open",
    "cancel_rebate",
    "serve_closed",
    "serve_open",
    "serve_rebate",
)
SOURCES = tuple((k, "explicit declared component118974:" + k) for k in st.COMPONENTS)


def q(v):
    return F(int(v["numerator"]), int(v["denominator"]))


def make_source(*, underflow_weight=None):
    # NEW native primitive fixture, adapted locally from the frozen constructor
    # pattern. No old helper imported or executed; no world, seed or tape.
    s = screen_spec(
        {"T": 2, "actions": 2, "n_spokes": 1, "capacity": 2, "capacity_support": [2]}
    )
    p = next(p for p in s.products if p.id == "A_OH0")
    m = next(m for m in s.markets if m.id == p.market)
    taste = replace(
        s.segments[0].classes[0],
        weight=1.0,
        alpha=1 / 60,
        carrier_pref={"A": 1.0, "B": 0.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    c = compile_market(
        replace(
            s,
            products=(replace(p, fare_floor=5.0, reference_fare=60.0, fare_cap=210.0),),
            markets=(m,),
            segments=(replace(s.segments[0], classes=(taste,), mixed=None),),
            demand=replace(
                s.demand,
                mode="single_request_bernoulli",
                arrival_probability=PeriodSchedule((0,), (0.5,)),
                arrival_probability_cap=1.0,
                market_weights=PeriodSchedule((0,), ((1.0,),)),
                segment_weights=PeriodSchedule((0,), ((1.0,),)),
                holiday_multiplier=1.0,
                exogenous_multipliers={},
                intensity=None,
            ),
            capacity_process=replace(
                s.capacity_process,
                events=(
                    CapacityEventSpec(
                        trigger={"hazard": {"probability": 0.25}},
                        effects=(("A_OH0", -1),),
                        once=True,
                    ),
                ),
            ),
            quotes=replace(s.quotes, requests=()),
            information=tuple(replace(a, requests=()) for a in s.information),
        )
    )
    initial = tuple(2 for _ in c.spec.legs)
    ledger = DeclaredBaseLedger(
        initial, (), tuple((0.0, 0.0, 0.0, 0, 0.0) for _ in c.spec.carriers), 0
    )
    paths = []
    for b0 in (False, True):
        for b1 in (False, True):
            installed = list(initial)
            fired = False
            ops = []
            for bit in (b0, b1):
                if bit and not fired:
                    installed[c.leg_index["A_OH0"]] -= 1
                    fired = True
                ops.append(BernoulliOpportunity(0, (60.0,), tuple(installed)))
            weight = (F(1, 4) if b0 else F(3, 4)) * (F(1, 4) if b1 else F(3, 4))
            paths.append(BernoulliFuturePath(f"bits{int(b0)}{int(b1)}", weight, ops))
    if underflow_weight is not None:
        # Pure declared prices, not native posted fares/cap acceptance.
        paths = [
            BernoulliFuturePath("normal", 1 - underflow_weight, paths[0].opportunities),
            BernoulliFuturePath(
                "underflow",
                underflow_weight,
                tuple(replace(o, prices=(1e6,)) for o in paths[0].opportunities),
            ),
        ]
    source = bernoulli_future_expectation(
        c,
        ledger,
        paths,
        start_period=0,
        mark_law=st.MARK_LAW,
        independent_future_private_marks=True,
        unreserved_base_ledger=True,
        conditioning="NEW118974: full fixed conditional program before private marks; not native physical/seed law",
        tree_row_limit=100000,
        conditional_call_limit=100000,
        ticket_limit=100000,
    )
    return c, source


def history_people(source, terminal):
    nodes = source.details["nodes"]
    people = []
    canceled = set()
    current = {t["record_id"] for t in terminal["ledger"]["tickets"]}
    for ni, ri in terminal["history"]:
        b = nodes[ni]["boundary"]["branches"][ri]
        canceled.update(t["record_id"] for t in b["canceled"])
        if b["input"][0] is not None:
            people.append(((nodes[ni]["period"], 0, b["input"][0]), b["new_ticket"]))
    return tuple(
        (
            key,
            ticket,
            ticket is not None and ticket["record_id"] in current,
            ticket is not None and ticket["record_id"] in canceled,
        )
        for key, ticket in people
    )


def programme(c, source, *, rare=F(1, 10**1800), open_mass=F()):
    histories = []
    for t in source.details["terminals"]:
        regime = (
            "early"
            if t["path"] in ("bits10", "bits11")
            else "late"
            if t["path"] == "bits01"
            else "none"
        )
        weights = {
            "early": (F(1, 2), F(), F(), F(1, 4), F(), F(1, 4)),
            "late": (F(1, 3), F(), F(), F(1, 3), F(), F(1, 3)),
            "none": (F(1, 6), F(), F(), F(1, 3), F(), F(1, 2)),
        }[regime]
        weights = tuple(
            w * (1 - open_mass) + (open_mass if i == 1 else 0)
            for i, w in enumerate(weights)
        )
        observations = []
        r = rare if regime == "none" else F()
        for signal, probability in (("rare", r), ("ordinary", 1 - r)):
            cells = []
            for name, weight in zip(CELLS, weights):
                financial = "open" not in name
                persons = []
                seat = 1 if name.startswith("serve") else 0
                for key, ticket, active, lost in history_people(source, t):
                    adjustments = []
                    service = None
                    net = F()
                    B = F(15)
                    W = F(4)
                    C = F()
                    if ticket is not None:
                        fare = F.from_float(ticket["fare"])
                        net = F() if lost else fare
                        if not active:
                            B = W = C = None
                        else:
                            service = bool(seat)
                            seat -= int(service)
                            B = F(240) if service else F()
                            W = F(7)
                            C = F() if service else F(19)
                            adjustment = (
                                F(-17)
                                if service and "rebate" in name
                                else -fare
                                if not service
                                else F()
                            )
                            adjustments = [
                                st.NetTicketAdjustment(
                                    f"extra:{key}",
                                    ticket["record_id"],
                                    c.spec.carriers[ticket["carrier"]].code,
                                    adjustment,
                                    "declared additional NET transfer118974",
                                )
                            ]
                            net += adjustment
                    persons.append(
                        st.PersonFutureMaterial(
                            key,
                            service,
                            "declared future service, not executed truth118974",
                            adjustments,
                            ContingentPayoff(B, net if financial else None, W, C),
                            SOURCES,
                            financial,
                            "declared financial tail118974",
                            True,
                            "declared material tail118974",
                        )
                    )
                cells.append(st.JointFutureCell(name, weight, persons))
            observations.append(st.HistoryObservation(signal, probability, cells))
        histories.append(st.HistoryFuture(t["path"], t["history"], observations))
    return st.SourceFutureProgramme(
        ("rare", "ordinary"),
        CELLS,
        histories,
        "explicit fare unit118974",
        "before explicit analyst history event118974",
        "full joint declared future conditional on complete supplied source118974",
        "NEW synthetic declaration118974",
    )


def event(source, *, offer=None, rare=False, unconditional=False, canceled=False):
    membership = []
    for t in source.details["terminals"]:
        nodes = source.details["nodes"]
        branches = [nodes[ni]["boundary"]["branches"][ri] for ni, ri in t["history"]]
        selected = (
            branches[0]["new_ticket"] is not None
            and branches[1]["input"][0] is not None
            and branches[1]["new_ticket"] is None
        )
        selected = selected and not any(b["canceled"] for b in branches)
        if offer is not None:
            selected = (
                selected
                and nodes[t["history"][1][0]]["boundary"]["supplied_new_birth_law"][
                    "global_net_sellable"
                ][0]
                is offer
            )
        if canceled:
            selected = any(b["canceled"] for b in branches)
        if unconditional:
            selected = True
        membership.append((selected, False if rare else selected))
    return st.SourceSelectionEvent(
        "explicit event118974",
        membership,
        "analyst source history and optional declared signal, not passenger knowledge",
        "synthetic event118974",
    )


@pytest.fixture(scope="module")
def bundle():
    c, s = make_source()
    return c, s, programme(c, s)


def run(bundle, **options):
    c, s, p = bundle
    result = st.condition_source_future(c, s, p, event(s, **options), row_limit=LIMIT)
    return st.source_future_transport_record(result)


def test_new_full_source_and_closed_form_selection(bundle):
    r = run(bundle)
    assert len(r["complete_source"]["details"]["nodes"]) == 16
    assert len(r["all_original_person_profiles"]) == 34
    assert len(r["all_atoms"]) == 34 * 2 * 6
    assert q(r["event_measure"]) == F(5, 64)
    assert q(r["raw_total_source_measure"]) == 1
    assert sum(q(a["conditional_mass"]) for a in r["conditional_atoms"]) == 1
    by_path = {
        p: sum(
            q(a["conditional_mass"]) for a in r["conditional_atoms"] if a["path"] == p
        )
        for p in ("bits00", "bits01", "bits10", "bits11")
    }
    assert by_path == {
        "bits00": F(9, 20),
        "bits01": F(3, 20),
        "bits10": F(3, 10),
        "bits11": F(1, 10),
    }
    # Material cancel=-15, serve=184, served rebate=201, includes outside11.
    expected = (
        F(2, 5) * (F(-15, 2) + F(184, 4) + F(201, 4))
        + F(3, 20) * F(-15 + 184 + 201, 3)
        + F(9, 20) * (F(-15, 6) + F(184, 3) + F(201, 2))
    )
    assert q(r["population_material"]["mean"]) == expected


def test_new_offer_partition_tower(bundle):
    all_r, no, yes = run(bundle), run(bundle, offer=False), run(bundle, offer=True)
    assert q(no["event_measure"]) == F(1, 32)
    assert q(yes["event_measure"]) == F(3, 64)
    assert q(all_r["event_measure"]) * q(all_r["population_material"]["mean"]) == sum(
        q(r["event_measure"]) * q(r["population_material"]["mean"]) for r in (no, yes)
    )


def test_new_rare_positive_and_exact_zero(bundle):
    r = run(bundle, rare=True)
    assert q(r["event_measure"]) == F(9, 256 * 10**1800)
    assert q(r["population_material"]["mean"]) == F(-15, 6) + F(184, 3) + F(201, 2)
    c, s, _ = bundle
    r = run((c, s, programme(c, s, rare=F())), rare=True)
    assert q(r["event_measure"]) == 0
    assert r["conditional_atoms"] is r["population_material"] is None
    assert len(r["all_atoms"]) == 408


def test_new_positive_tiny_open_financial_tail(bundle):
    c, s, _ = bundle
    r = run((c, s, programme(c, s, open_mass=F(1, 10**1700))))
    assert r["population_material"]["mean"] is r["population_payment"]["mean"] is None
    assert q(r["population_material"]["unknown_mass"]) == F(1, 10**1700)


def test_new_unreserved_original_and_unconditional_unknown(bundle):
    r = run(bundle, unconditional=True)
    assert r["population_material"]["mean"] is None
    assert any(
        p["original_ticket"] is not None and p["current_ticket"] is None
        for ps in r["all_original_person_profiles"]
        for p in ps
    )
    assert any(not ps for ps in r["all_original_person_profiles"])
    assert any(
        p["original_ticket"] is None
        for ps in r["all_original_person_profiles"]
        for p in ps
    )
    assert r["lifecycle_utility_CS_total_denial_misallocation"] is None
    assert not r[
        "native_source_probability_ledger_actor_RNG_or_service_executor_called"
    ]


@pytest.mark.parametrize("weight", [F(), F(1, 10**1900)])
def test_new_needed_underflow_never_salvaged(weight):
    c, s = make_source(underflow_weight=weight)
    r = run((c, s, programme(c, s)), unconditional=True)
    assert r["source_public_joint_known"] is (weight == 0)
    if weight:
        assert (
            r["event_measure"]
            is r["conditional_atoms"]
            is r["population_material"]
            is None
        )
    assert r["raw_event_measure_diagnostic_only"] is not None
    assert any(
        a["path"] == "underflow" and q(a["raw_joint_mass"]) == 0 for a in r["all_atoms"]
    )


def changed_source(s, mutate):
    data = _encoded(s)
    mutate(data)
    # Result codec already encoded; ownership constructor deliberately not a seal.
    return BernoulliFutureResult(
        data["joint"], data["expected_counts"], data["details"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["details"].__setitem__("nodes", d["details"]["nodes"][:-1]),
        lambda d: d["details"].__setitem__("terminals", d["details"]["terminals"][:-1]),
        lambda d: d["details"]["nodes"][0]["boundary"].__setitem__(
            "branches", d["details"]["nodes"][0]["boundary"]["branches"][:-1]
        ),
        lambda d: d["details"]["nodes"][0].__setitem__("structurally_needed", False),
        lambda d: d["details"].__setitem__("needed_unknown_nodes", (0,)),
        lambda d: d.__setitem__("joint", ()),
        lambda d: d["details"]["initial_declared_ledger"].__setitem__(
            "next_record_id", 1
        ),
        lambda d: d["details"]["paths"][0]["weight"].__setitem__("numerator", "2"),
    ],
)
def test_new_complete_source_refusals(bundle, mutation):
    c, s, p = bundle
    bad = changed_source(s, mutation)
    with pytest.raises((ValueError, KeyError)):
        st.condition_source_future(c, bad, p, event(s), row_limit=LIMIT)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(p, histories=p.histories[:-1]),
        lambda p: replace(p, signals=("ordinary", "rare")),
        lambda p: replace(p, cells=p.cells[:-1]),
        lambda p: replace(
            p,
            histories=(
                replace(p.histories[0], observations=p.histories[0].observations[:-1]),
                *p.histories[1:],
            ),
        ),
        lambda p: replace(p, payoff_unit=""),
    ],
)
def test_new_complete_programme_refusals(bundle, change):
    c, s, p = bundle
    with pytest.raises(ValueError):
        st.condition_source_future(c, s, change(p), event(s), row_limit=LIMIT)


@pytest.mark.parametrize(
    "change",
    [
        lambda e: replace(e, membership=e.membership[:-1]),
        lambda e: replace(e, membership=((1, False), *e.membership[1:])),
        lambda e: replace(e, membership=((False,), *e.membership[1:])),
    ],
)
def test_new_exact_event_support_refusals(bundle, change):
    c, s, p = bundle
    with pytest.raises(ValueError):
        st.condition_source_future(c, s, p, change(event(s)), row_limit=LIMIT)


def alter_first_person(p, change):
    hs = list(p.histories)
    hi = next(i for i, h in enumerate(hs) if h.observations[0].cells[0].persons)
    h = hs[hi]
    obs = h.observations[0]
    cell = obs.cells[0]
    cell = replace(cell, persons=(change(cell.persons[0]), *cell.persons[1:]))
    obs = replace(obs, cells=(cell, *obs.cells[1:]))
    hs[hi] = replace(h, observations=(obs, *h.observations[1:]))
    return replace(p, histories=hs)


@pytest.mark.parametrize(
    "change",
    [
        lambda t: replace(t, person_key=(99, 0, 0)),
        lambda t: replace(t, financial_closed=1),
        lambda t: replace(t, material_source=""),
        lambda t: replace(t, component_sources=t.component_sources[:-1]),
        lambda t: replace(t, payoff=replace(t.payoff, net_ticket_payment=F(1000))),
        lambda t: replace(
            t,
            adjustments=(st.NetTicketAdjustment("wrong", 99, "A", F(-1), "declared"),),
        ),
    ],
)
def test_new_all_person_payment_and_closure_refusals(bundle, change):
    c, s, p = bundle
    with pytest.raises(ValueError):
        st.condition_source_future(
            c, s, alter_first_person(p, change), event(s), row_limit=LIMIT
        )


def test_new_preflight_limit_and_failure_ownership(bundle, monkeypatch):
    c, s, p = bundle
    before = _encoded((s, p))
    e = event(s)
    with pytest.raises(st.SourceTransportLimit):
        st.condition_source_future(c, s, p, e, row_limit=1)
    assert _encoded((s, p)) == before

    def fail(*args):
        raise RuntimeError("injected pure projection failure118974")

    monkeypatch.setattr(st, "_source", fail)
    with pytest.raises(RuntimeError, match="injected"):
        st.condition_source_future(c, s, p, e, row_limit=LIMIT)
    assert _encoded((s, p)) == before


def test_new_owned_result_and_zero_unknown_rows(bundle):
    c, s, p = bundle
    before = _encoded((s, p))
    e = event(s)
    result = st.condition_source_future(c, s, p, e, row_limit=LIMIT)
    record = st.source_future_transport_record(result)
    assert any(
        a["population_material"] is None and q(a["raw_joint_mass"]) == 0
        for a in record["all_atoms"]
    )
    record["all_atoms"][0]["path"] = "mutated owned export"
    assert _encoded((s, p)) == before
    assert st.source_future_transport_record(result) != record


def test_new_no_arrival_has_no_people_not_outside(bundle):
    c, s, p = bundle
    e = event(s)
    membership = tuple((not history_people(s, t),) * 2 for t in s.details["terminals"])
    r = st.source_future_transport_record(
        st.condition_source_future(
            c, s, p, replace(e, membership=membership), row_limit=LIMIT
        )
    )
    assert q(r["event_measure"]) == F(1, 4)
    assert q(r["population_material"]["mean"]) == 0
    assert all(not a["persons"] for a in r["conditional_atoms"])


@pytest.mark.parametrize("weight", [F(2), F(-1), 1])
def test_new_all_zero_history_future_distributions_validated(bundle, weight):
    c, s, p = bundle
    h = p.histories[0]
    o = h.observations[0]
    cell = o.cells[0]
    # This rare signal is unreachable here; its declared full future still validates.
    bad = replace(
        p,
        histories=(
            replace(
                h,
                observations=(
                    replace(o, cells=(replace(cell, probability=weight), *o.cells[1:])),
                    *h.observations[1:],
                ),
            ),
            *p.histories[1:],
        ),
    )
    with pytest.raises(ValueError):
        st.condition_source_future(c, s, bad, event(s), row_limit=LIMIT)


def test_new_pure_reader_no_native_forward_or_global_rng(bundle, monkeypatch):
    import numpy as np

    from airline_market.sim import (
        bernoulli_boundary,
        bernoulli_future,
        bernoulli_person,
    )

    c, s, p = bundle
    state = np.random.get_state()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected native producer forward")

    monkeypatch.setattr(bernoulli_future, "bernoulli_future_expectation", fail)
    monkeypatch.setattr(bernoulli_boundary, "resolve_bernoulli_boundary", fail)
    monkeypatch.setattr(bernoulli_person, "native_bernoulli_person_law", fail)
    st.condition_source_future(c, s, p, event(s), row_limit=LIMIT)
    after = np.random.get_state()
    assert (
        state[0] == after[0]
        and np.array_equal(state[1], after[1])
        and state[2:] == after[2:]
    )
