"""A.10/A.5/B0/V9/B4: NEW124974 units; no old fixture/source/case loader."""

import copy
import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim import bernoulli_service as bs
from airline_market.sim import branch_financial as bf
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliOpportunity,
    bernoulli_future_expectation,
)
from airline_market.sim.bernoulli_person import MARK_LAW
from airline_market.sim.current_itinerary_service import LegOperation
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

CLOSE = datetime(2027, 10, 12, 0, 2, tzinfo=timezone.utc)
ASOF = CLOSE + timedelta(days=2)
NAMES = ("none", "refund", "open", "unknown", "credit", "nan")


def q(value):
    return (
        None if value is None else F(int(value["numerator"]), int(value["denominator"]))
    )


def fixture(
    *, T=1, alternatives=False, underflow=False, arrival=0.1875, nan_cash=False
):
    s = screen_spec(
        {
            "T": max(2, T),
            "actions": 2,
            "n_spokes": 1,
            "capacity": 2,
            "capacity_support": [2],
        }
    )
    p = next(p for p in s.products if p.id == "A_OH0")
    market = next(m for m in s.markets if m.id == p.market)
    taste = replace(
        s.segments[0].classes[0],
        weight=1.0,
        alpha=1 / 80,
        carrier_pref={"A": 1.0, "B": 1.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    products = (replace(p, fare_floor=1.0, reference_fare=80.0, fare_cap=500.0),)
    if alternatives:
        products = (*products, replace(products[0], id=products[0].id + "_new124974"))
    c = compile_market(
        replace(
            s,
            products=products,
            markets=(market,),
            segments=(replace(s.segments[0], classes=(taste,), mixed=None),),
            demand=replace(
                s.demand,
                mode="single_request_bernoulli",
                arrival_probability=PeriodSchedule((0,), (arrival,)),
                arrival_probability_cap=1.0,
                market_weights=PeriodSchedule((0,), ((1.0,),)),
                segment_weights=PeriodSchedule((0,), ((1.0,),)),
                holiday_multiplier=1.0,
                exogenous_multipliers={},
                intensity=None,
            ),
            quotes=replace(s.quotes, requests=()),
            information=tuple(replace(a, requests=()) for a in s.information),
        )
    )
    full = tuple(2 for _ in c.spec.legs)
    accounts = tuple(
        (float("nan") if nan_cash and i == 0 else 0.0, 0.0, 0.0, 0, 0.0)
        for i, _ in enumerate(c.spec.carriers)
    )
    ledger = DeclaredBaseLedger(full, (), accounts, 0)
    op = BernoulliOpportunity(
        0, tuple(1e6 if underflow else 80.0 for _ in c.spec.products), full
    )
    paths = (BernoulliFuturePath("keep124974", F(1), tuple(op for _ in range(T))),)
    if T == 2:
        low = list(full)
        low[c.leg_index[c.spec.products[0].legs[0]]] = 1
        paths = (
            BernoulliFuturePath("keep124974", F(2, 3), (op, op)),
            BernoulliFuturePath(
                "loss124974", F(1, 3), (op, replace(op, installed_after=tuple(low)))
            ),
        )
    source = bernoulli_future_expectation(
        c,
        ledger,
        paths,
        start_period=0,
        conditioning="NEW124974 explicit finite source; not seed/physical law/knowledge authentication",
        mark_law=MARK_LAW,
        independent_future_private_marks=True,
        unreserved_base_ledger=True,
        tree_row_limit=100000,
        conditional_call_limit=10000,
        ticket_limit=T,
    )
    clock = bs.BranchServiceClock(
        CLOSE,
        tuple((leg.id, CLOSE + timedelta(hours=2)) for leg in c.spec.legs),
        "NEW124974 declared UTC projection",
    )
    catalogue = ((None, None, None, None),) + tuple(
        (0, 0, 0, j) for j in (*c.market_products[0], len(c.spec.products))
    )
    cases = []
    for path in paths:
        for branches in itertools.product(catalogue, repeat=T):
            products_bought = tuple(
                row[3]
                for row in branches
                if row[0] is not None and row[3] != len(c.spec.products)
            )
            for active in itertools.product((False, True), repeat=len(products_bought)):
                targets = tuple(
                    (rid, c.spec.products[j].legs[0])
                    for rid, j in enumerate(products_bought)
                    if active[rid]
                )
                programmes = []
                for name, probability in zip(
                    ("board124974", "cancel124974", "zero124974"),
                    (F(3, 4), F(1, 4), F()),
                ):
                    cancelled = name == "cancel124974"
                    programmes.append(
                        bs.BranchServiceProgramme(
                            name,
                            probability,
                            tuple(
                                LegOperation(
                                    leg.id,
                                    "cancelled" if cancelled else "operates",
                                    0 if name == "zero124974" else 2,
                                    None if cancelled else CLOSE + timedelta(hours=2),
                                    None if cancelled else CLOSE + timedelta(hours=3),
                                    CLOSE + timedelta(hours=1) if cancelled else None,
                                )
                                for leg in c.spec.legs
                            ),
                            tuple((rid, leg, True) for rid, leg in targets),
                            tuple(
                                (
                                    leg.id,
                                    tuple(rid for rid, l in targets if l == leg.id),
                                )
                                for leg in c.spec.legs
                            ),
                            "NEW124974 full direct programme",
                            "NEW124974 unit declaration",
                        )
                    )
                cases.append(
                    bs.BranchServiceCase(path.name, branches, active, programmes)
                )
    cases = tuple(cases)
    service = bs.bernoulli_service_programmes(
        c,
        source,
        clock,
        ("board124974", "cancel124974", "zero124974"),
        cases,
        definition="NEW124974 service source",
        support_limit=10**8,
    )
    record = bs.bernoulli_service_record(service)
    cells = []
    for i, atom in enumerate(record["all_atoms"]):
        people = record["all_original_person_profiles"][atom["history_index"]]
        status = dict(atom["service"]["current_service"])
        programmes = []
        for name in NAMES:
            transfers = []
            for person in people:
                rid = person["original_ticket"]
                if rid is None:
                    continue
                value = q(person["original_fare"])
                if (
                    name == "refund"
                    and person["current_ticket"] is not None
                    and status[person["person_key"]] is False
                ):
                    transfers.append(
                        bf.BranchTicketTransfer(
                            f"refund-{rid}",
                            person["person_key"],
                            rid,
                            person["carrier"],
                            "additional_refund",
                            float(value),
                            CLOSE + timedelta(hours=10),
                            CLOSE + timedelta(hours=10, seconds=1),
                            "NEW124974 explicit additional non-delivery settlement",
                            "Not generated by reader",
                        )
                    )
                if name in ("unknown", "credit", "nan"):
                    amount = (
                        None
                        if name == "unknown"
                        else -3.0
                        if name == "credit"
                        else float("nan")
                    )
                    transfers.append(
                        bf.BranchTicketTransfer(
                            f"settlement-{rid}",
                            person["person_key"],
                            rid,
                            person["carrier"],
                            "additional_payment",
                            amount,
                            CLOSE + timedelta(hours=10),
                            CLOSE + timedelta(hours=10, seconds=1),
                            "NEW124974 additional settlement",
                            "Declared separately from native refunds",
                        )
                    )
                if name == "credit":
                    transfers.append(
                        bf.BranchTicketTransfer(
                            f"signed-zero-{rid}",
                            person["person_key"],
                            rid,
                            person["carrier"],
                            "additional_refund",
                            -0.0,
                            CLOSE + timedelta(hours=10),
                            CLOSE + timedelta(hours=10, seconds=1),
                            "NEW124974 signed zero",
                            "Explicit no automatic compensation",
                        )
                    )
            followup = tuple(
                bf.BranchFinancialFollowup(
                    p["person_key"],
                    ASOF,
                    name != "open",
                    "NEW124974 independent financial follow-up",
                )
                for p in people
            )
            programmes.append(
                bf.BranchSettlement(
                    name, transfers, followup, "NEW124974 explicit settlement support"
                )
            )
        cells.append(bf.BranchFinancialCell(i, programmes))
    eps = F(1, 10**1000)
    weights = (
        ("none", (F(1), F(), F(), F(), F(), F())),
        ("refund", (F(), F(1), F(), F(), F(), F())),
        ("tiny_open", (1 - eps, F(), eps, F(), F(), F())),
        ("tiny_unknown", (1 - eps, F(), F(), eps, F(), F())),
        ("credit", (F(), F(), F(), F(), F(1), F())),
    )
    laws = tuple(
        bf.BranchFinancialLaw(
            name,
            tuple(row for _ in cells),
            "NEW124974 declared conditional financial law",
        )
        for name, row in weights
    )
    purchased = tuple(
        any(
            p["original_ticket"] is not None
            for p in record["all_original_person_profiles"][a["history_index"]]
        )
        for a in record["all_atoms"]
    )
    events = (
        bf.BranchFinancialEvent(
            "all", tuple(True for _ in cells), "ALL original source/service support"
        ),
        bf.BranchFinancialEvent(
            "purchase",
            purchased,
            "Explicit analyst original-purchase event, not inferred prior",
        ),
        bf.BranchFinancialEvent(
            "zero", tuple(False for _ in cells), "Exact-zero event remains explicit"
        ),
    )
    return c, source, clock, cases, service, NAMES, tuple(cells), laws, events


def run(base, **kwargs):
    return bf.branch_financial_record(
        bf.branch_financial_laws(
            *base,
            financial_asof=ASOF,
            unit="declared-native-ticket-unit-124974",
            definition="NEW124974 actual-atom supplied reader, not prospective registration",
            support_limit=kwargs.pop("support_limit", 10**9),
            **kwargs,
        )
    )


def summary(record, law, event="all"):
    return next(
        s for s in record["summaries"] if s["law"] == law and s["event"] == event
    )


def test_new_closed_form_full_joint_and_conditional_events():
    r = run(fixture())
    assert q(summary(r, "none")["net_payment_integral"]) == F(15, 2)
    assert q(summary(r, "refund")["net_payment_integral"]) == F(45, 8)
    assert q(summary(r, "credit")["net_payment_integral"]) == F(231, 32)
    assert (
        q(summary(r, "none", "purchase")["raw_measure_conditional_payment_ratio"]) == 80
    )
    assert (
        q(summary(r, "refund", "purchase")["raw_measure_conditional_payment_ratio"])
        == 60
    )
    assert q(summary(r, "none", "purchase")["raw_event_evidence"]) == F(3, 32)
    assert summary(r, "none", "zero")["raw_measure_conditional_payment_ratio"] is None
    assert q(summary(r, "none", "zero")["net_payment_integral"]) == 0
    for name in ("none", "refund", "tiny_open", "tiny_unknown", "credit"):
        assert sum(
            (q(x["raw_joint_mass"]) for x in r["raw_law_rows"] if x["law"] == name), F()
        ) == q(r["raw_source_measure"])
    assert r["material_utility_CS_denial_misallocation_penalty_incidence"] is None
    assert not r["native_or_service_producer_RNG_actor_book_resolve_called"]
    assert not r["pre_source_registration_or_financial_service_truth_authenticated"]


def test_new_tiny_open_outside_and_unknown_rows_not_zero_filled():
    r = run(fixture())
    for name in ("tiny_open", "tiny_unknown"):
        s = summary(r, name)
        assert (
            s["net_payment_integral"]
            is s["raw_measure_conditional_payment_ratio"]
            is None
        )
        assert (
            q(s["positive_unknown_mass"]) > 0
            and float(q(s["positive_unknown_mass"])) == 0
        )
    outside = [
        p
        for row in r["payment_reports"]
        if row["programme"] == "open"
        for p in row["all_original_people"]
        if p["original_ticket"] is None
    ]
    assert outside and all(
        q(p["recorded_net"]) == 0 and p["final_net"] is None for p in outside
    )
    assert any(
        row["population_final_net"] is None
        for row in r["payment_reports"]
        if row["programme"] == "nan"
    )
    zeros = [
        x
        for x in r["raw_law_rows"]
        if x["programme"] == "unknown" and x["law"] == "none"
    ]
    assert zeros and all(q(x["raw_joint_mass"]) == 0 for x in zeros)


def test_new_native_source_refund_once_and_cancelled_ticket_additional_credit():
    r = run(fixture(T=2))
    cancelled = [
        p
        for row in r["payment_reports"]
        if row["programme"] == "none"
        for p in row["all_original_people"]
        if p["original_ticket"] is not None and p["current_ticket"] is None
    ]
    assert cancelled and all(
        q(p["native_refund_once"]) == 80 and q(p["final_net"]) == 0 for p in cancelled
    )
    credit = [
        p
        for row in r["payment_reports"]
        if row["programme"] == "credit"
        for p in row["all_original_people"]
        if p["original_ticket"] is not None and p["current_ticket"] is None
    ]
    assert credit and all(
        q(p["native_refund_once"]) == 80 and q(p["final_net"]) == -3 for p in credit
    )
    payload = r["cells"]
    assert any(
        e["amount"]["binary64_bits"] == "8000000000000000"
        for cell in payload
        for p in cell["programmes"]
        for e in p["transfers"]
        if e["amount"] is not None
    )


def test_new_upstream_underflow_whole_public_unknown_but_raw_retained():
    r = run(fixture(underflow=True))
    assert (
        not r["source_public_joint_known"]
        and r["public_law_rows"] is None
        and r["raw_law_rows"]
    )
    assert all(
        s["net_payment_integral"] is s["raw_measure_conditional_payment_ratio"] is None
        for s in r["summaries"]
    )


def test_new_raw_mass_residual_and_nonfinite_carrier_context_retained():
    r = run(fixture(alternatives=True, nan_cash=True))
    assert q(r["raw_source_residual"]) != 0
    assert (
        r["complete_native_source"]["details"]["initial_declared_ledger"]["accounts"][
            0
        ][0]["exact"]
        is None
    )
    for law in r["laws"]:
        assert sum(
            (
                q(row["raw_joint_mass"])
                for row in r["raw_law_rows"]
                if row["law"] == law["name"]
            ),
            F(),
        ) == q(r["raw_source_measure"])


def test_new_empty_source_and_source_global_rng_unchanged_owned_exports():
    base = fixture(arrival=0.0)
    before = _encoded(base[:5])
    state = np.random.get_state()
    r = run(base)
    assert q(summary(r, "none")["net_payment_integral"]) == 0
    assert _encoded(base[:5]) == before
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))
    typed = bf.branch_financial_laws(
        *base,
        financial_asof=ASOF,
        unit="unit124974",
        definition="NEW124974 owned",
        support_limit=10**9,
    )
    exported = bf.branch_financial_record(typed)
    exported["summaries"] = ()
    assert bf.branch_financial_record(typed)["summaries"]
    with pytest.raises((TypeError, AttributeError)):
        typed.analyst["new"] = 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing_cell",
        "reordered_cell",
        "missing_programme",
        "missing_followup",
        "bad_closure",
        "wrong_carrier",
        "duplicate_transfer",
        "outside_transfer",
        "late_record",
        "bad_kind",
    ],
)
def test_new_complete_support_and_transfer_refusals(kind):
    base = list(fixture())
    cells = list(base[6])
    ix = next(i for i, c in enumerate(cells) if c.programmes[4].transfers)
    cell = cells[ix]
    programmes = list(cell.programmes)
    p = programmes[4]
    if kind == "missing_cell":
        cells.pop()
    elif kind == "reordered_cell":
        cells[0], cells[1] = cells[1], cells[0]
    elif kind == "missing_programme":
        cells[ix] = replace(cell, programmes=cell.programmes[:-1])
    elif kind == "missing_followup":
        programmes[4] = replace(p, followup=())
    elif kind == "bad_closure":
        programmes[4] = replace(p, followup=(replace(p.followup[0], closed=1),))
    elif kind == "wrong_carrier":
        programmes[4] = replace(
            p, transfers=(replace(p.transfers[0], carrier="B"), *p.transfers[1:])
        )
    elif kind == "duplicate_transfer":
        programmes[4] = replace(p, transfers=(*p.transfers, p.transfers[-1]))
    elif kind == "outside_transfer":
        outside_ix = next(
            i
            for i, c in enumerate(cells)
            if c.programmes[0].followup and not c.programmes[4].transfers
        )
        oc = cells[outside_ix]
        op = oc.programmes[0]
        fake = replace(p.transfers[0], person_key=op.followup[0].person_key)
        cells[outside_ix] = replace(
            oc, programmes=(replace(op, transfers=(fake,)), *oc.programmes[1:])
        )
    elif kind == "late_record":
        programmes[4] = replace(
            p,
            transfers=(
                replace(p.transfers[0], recorded=ASOF + timedelta(seconds=1)),
                *p.transfers[1:],
            ),
        )
    else:
        programmes[4] = replace(
            p,
            transfers=(
                replace(p.transfers[0], kind="implicit_refund"),
                *p.transfers[1:],
            ),
        )
    if kind not in (
        "missing_cell",
        "reordered_cell",
        "missing_programme",
        "outside_transfer",
    ):
        cells[ix] = replace(cell, programmes=programmes)
    base[6] = cells
    with pytest.raises(ValueError):
        run(base)


@pytest.mark.parametrize(
    "kind",
    [
        "float_probability",
        "nonunit_law",
        "omitted_probability",
        "bad_mask",
        "omitted_mask",
    ],
)
def test_new_exact_full_law_and_event_refusals(kind):
    base = list(fixture())
    if kind in ("bad_mask", "omitted_mask"):
        e = base[8][0]
        include = (1, *e.include[1:]) if kind == "bad_mask" else e.include[:-1]
        base[8] = (replace(e, include=include), *base[8][1:])
    else:
        law = base[7][0]
        row = list(law.probabilities[0])
        if kind == "float_probability":
            row[0] = 1.0
        elif kind == "nonunit_law":
            row[0] = F(1, 2)
        else:
            row.pop()
        base[7] = (
            replace(law, probabilities=(tuple(row), *law.probabilities[1:])),
            *base[7][1:],
        )
    with pytest.raises(ValueError):
        run(base)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_step",
        "wrong_decision",
        "board_as_delivery",
        "wrong_joint",
        "wrong_count",
        "missing_atom",
    ],
)
def test_new_full_supplied_service_witness_refusals(kind):
    base = list(fixture())
    record = copy.deepcopy(bs.bernoulli_service_record(base[4]))
    i = next(
        i
        for i, a in enumerate(record["all_atoms"])
        if a["service"]["delivered_current_target_count"]
    )
    atoms = list(record["all_atoms"])
    a = atoms[i]
    if kind == "missing_step":
        a["service"]["steps"] = a["service"]["steps"][:-1]
    elif kind in ("wrong_decision", "board_as_delivery"):
        j = next(
            j
            for j, s in enumerate(a["service"]["steps"])
            if s["kind"] == "departure" and s["decisions"]
        )
        steps = list(a["service"]["steps"])
        steps[j]["decisions"] = (
            (
                steps[j]["decisions"][0][0],
                "no_seat" if kind == "wrong_decision" else "delivered_at_arrival",
            ),
        )
        a["service"]["steps"] = tuple(steps)
    elif kind == "wrong_joint":
        a["raw_joint_mass"] = _encoded(F(1, 2))
    elif kind == "wrong_count":
        a["service"]["delivered_current_target_count"] = 2
    else:
        atoms.pop()
    record["all_atoms"] = tuple(atoms)
    base[4] = bs.BernoulliServiceResult(record)
    with pytest.raises(ValueError):
        run(base)


def test_new_complete_output_bound_before_payment_arithmetic(monkeypatch):
    base = fixture()
    result = run(base)
    bound = result["all_support_bound"]
    assert bf._units(_encoded(result)) <= bound

    def forbidden(*args, **kwargs):
        raise AssertionError("payment arithmetic reached before support refusal")

    monkeypatch.setattr(bf, "_payments", forbidden)
    with pytest.raises(bf.BranchFinancialLimit):
        run(base, support_limit=bound - 1)
    monkeypatch.setattr(bf, "_service", forbidden)
    with pytest.raises(bf.BranchFinancialLimit):
        run(base, support_limit=1)


def test_new_no_upstream_execution_and_injected_failure(monkeypatch):
    base = fixture()
    before = _encoded(base)
    import airline_market.sim.bernoulli_future as native

    def forbidden(*args, **kwargs):
        raise AssertionError("upstream producer called")

    monkeypatch.setattr(native, "bernoulli_future_expectation", forbidden)
    monkeypatch.setattr(bs, "_execute", forbidden)
    r = run(base)
    assert r["supplied_service_witness_checked_not_executed"]
    original = bf._payments
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("NEW124974 injected pure failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(bf, "_payments", broken)
    with pytest.raises(RuntimeError, match="injected"):
        run(base)
    assert len(calls) == 2 and _encoded(base) == before


@pytest.mark.parametrize("key", [(False, 0, 0), (-1, 0, 0), (0.0, 0, 0)])
def test_new_key_types_and_naive_clock_refused(key):
    with pytest.raises(ValueError):
        bf.BranchFinancialFollowup(key, ASOF, True, "new124974")


def test_new_naive_and_early_financial_asof_refused():
    with pytest.raises(ValueError):
        bf.BranchFinancialFollowup(
            (0, 0, 0), ASOF.replace(tzinfo=None), True, "new124974"
        )
    base = fixture()
    with pytest.raises(ValueError):
        bf.branch_financial_laws(
            *base,
            financial_asof=CLOSE,
            unit="unit124974",
            definition="new124974",
            support_limit=10**9,
        )


def test_new_opaque_transfer_iterator_refused():
    base = fixture()
    p = next(c.programmes[4] for c in base[6] if c.programmes[4].transfers)
    with pytest.raises(ValueError, match="tuple/list"):
        replace(p, transfers=iter(p.transfers))


@pytest.mark.parametrize("field", (6, 7, 8))
def test_new_opaque_cell_iterator_refused(field):
    base = list(fixture())
    base[field] = iter(base[field])
    with pytest.raises(ValueError, match="tuple/list"):
        run(base)
