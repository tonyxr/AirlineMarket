"""A.10/A.5/A.14/B0/V9: complete NEW original-person native context join.

Captured ordinary/earlier own/proof views, not full received knowledge, final
re-entry menus, inferred type/law, passenger utility or changed native behavior.
"""

from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc
from airline_market.sim.own_history_information import (
    OwnInformationPeriod,
    OwnNativeFact,
    OwnReentryRow,
    OwnReentryView,
    own_reentry_information_view,
)
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.proof_information import (
    NativePresentationFact,
    NativePresentationRow,
    ProofInformationPeriod,
    ProofReentryRow,
    ProofReentryView,
    proof_information_view,
)
from airline_market.sim.received_information import (
    _key,
    _text,
    _units,
    received_information_record,
)

RULE = "complete-captured-native-original-person-contexts-v1"
KINDS = {
    "native_ordinary_purchase",
    "native_ticket_cancelled",
    "native_reentry_purchase",
    "native_reentry_outside",
}


@dataclass(frozen=True)
class NativeDecisionContext:
    analyst_original_key: tuple
    analyst_root: int | None
    person_decision: int
    view: object

    def __post_init__(self):
        object.__setattr__(
            self, "analyst_original_key", _key(self.analyst_original_key)
        )
        object.__setattr__(self, "view", deep_freeze(self.view))


@dataclass(frozen=True)
class NativeContextPeriod:
    period: int
    reentry_contexts: tuple
    ordinary_contexts: tuple


@dataclass(frozen=True)
class NativeInformationContexts:
    analyst_context: object
    periods: tuple
    people: tuple
    definition: str
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst_context", deep_freeze(self.analyst_context))
        object.__setattr__(self, "periods", tuple(self.periods))
        object.__setattr__(self, "people", deep_freeze(self.people))


class NativeInformationContextLimit(RuntimeError):
    """ALL supplied source and proposed output structural support exceeds limit."""


def _facts(facts, tickets, original, products):
    """Validate complete NEW prospective ticket/fact stream, not old reconstruction."""
    if not facts or not tickets or tickets[0] != original["original_record"]:
        raise ValueError("complete booked root facts and original ticket required")
    bought, cancelled, searches = {}, set(), []
    previous = -1
    for i, f in enumerate(facts):
        if (
            type(f) is not OwnNativeFact
            or type(f.ordinal) is not int
            or f.ordinal != i
            or f.kind not in KINDS
        ):
            raise ValueError("complete ordered typed native fact support required")
        if type(f.native_period) is not int or f.native_period < previous:
            raise ValueError("nondecreasing native fact periods required")
        previous = f.native_period
        if f.kind in {"native_ordinary_purchase", "native_reentry_purchase"}:
            local = len(bought)
            if (
                type(f.person_ticket) is not int
                or f.person_ticket != local
                or local >= len(tickets)
                or f.product_id not in products
                or type(f.fare) is not float
            ):
                raise ValueError(
                    "complete unique local purchase ticket/product/fare required"
                )
            if local == 0:
                chosen = original["native_row"]["chosen"]
                if (
                    f.kind != "native_ordinary_purchase"
                    or f.native_period != original["key"][0]
                    or f.product_id != tuple(products)[chosen]
                    or f.stage is not None
                ):
                    raise ValueError(
                        "first native fact must match original ordinary purchase"
                    )
            elif f.kind != "native_reentry_purchase" or f.stage not in {
                "own",
                "interline",
            }:
                raise ValueError(
                    "replacement requires actual native re-entry purchase fact"
                )
            if local and local - 1 not in cancelled:
                raise ValueError(
                    "replacement cannot replace an uncancelled active ticket"
                )
            bought[local] = f
        elif f.kind == "native_ticket_cancelled":
            p = bought.get(f.person_ticket)
            if (
                type(f.person_ticket) is not int
                or p is None
                or f.person_ticket != len(bought) - 1
                or f.person_ticket in cancelled
                or (f.product_id, _encode(f.fare)) != (p.product_id, _encode(p.fare))
                or f.stage is not None
            ):
                raise ValueError(
                    "cancellation must match one earlier same-person ticket"
                )
            cancelled.add(f.person_ticket)
        else:
            if len(bought) - 1 not in cancelled:
                raise ValueError("outside search requires no active current ticket")
            if any(
                v is not None for v in (f.person_ticket, f.product_id, f.fare, f.stage)
            ):
                raise ValueError(
                    "outside search cannot invent ticket/product/fare/stage"
                )
        if f.kind.startswith("native_reentry_"):
            if f.native_period <= original["key"][0] or not cancelled:
                raise ValueError(
                    "native search must follow actual original cancellation"
                )
            searches.append(f)
    if len(bought) != len(tickets) or len({f.native_period for f in searches}) != len(
        searches
    ):
        raise ValueError(
            "full ticket support and at most one same-root native attempt per period required"
        )
    return tuple(searches)


def native_information_contexts(
    journal,
    own_periods,
    proof_periods,
    final_histories,
    presentation_calls,
    *,
    definition,
    row_limit,
):
    """Join supplied NEW full support; invoke no engine/account/timing producer.

    Per-native-period re-entry precedes ordinary admission. Within own facts and
    proof calls only their captured sequence is retained: no cross-stream instant
    order, actual reception, missing stage menu/type or probability is invented.
    """
    _text(definition)
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive all-support infrastructure limit required")
    source = received_information_record(journal)
    context = source["analyst_context"]
    n = context["completed_source_periods"]
    own_periods, proof_periods = tuple(own_periods), tuple(proof_periods)
    final_histories, presentation_calls = (
        tuple(final_histories),
        tuple(presentation_calls),
    )
    if len(own_periods) != n or len(proof_periods) != n:
        raise ValueError("ALL native periods including empty periods required")
    ordinary = tuple(r for p in context["periods"] for r in p["rows"])
    by_key = {tuple(r["analyst_person"]["key"]): r for r in ordinary}
    roots = {
        r["analyst_person"]["original_record"]: tuple(r["analyst_person"]["key"])
        for r in ordinary
        if r["analyst_person"]["original_record"] is not None
    }
    if len(roots) != sum(
        r["analyst_person"]["original_record"] is not None for r in ordinary
    ):
        raise ValueError("distinct original booked roots required")
    products = {p["id"]: p for p in context["native_spec"]["products"]}
    histories, tickets_by_root, expected_searches = {}, {}, {}
    all_tickets = set()
    if len(final_histories) != len(roots):
        raise ValueError(
            "EVERY originally booked root final prospective history required"
        )
    for row in final_histories:
        if len(row) != 4:
            raise ValueError("complete root/facts/tickets/original-key row required")
        root, facts, tickets, key = row
        key, facts, tickets = _key(key), tuple(facts), tuple(tickets)
        if (
            type(root) is not int
            or root not in roots
            or roots[root] != key
            or root in histories
        ):
            raise ValueError("same source original-person root association required")
        if any(type(t) is not int or t < 0 or t in all_tickets for t in tickets) or len(
            set(tickets)
        ) != len(tickets):
            raise ValueError("global tickets cannot transfer between original people")
        all_tickets.update(tickets)
        searches = _facts(facts, tickets, by_key[key]["analyst_person"], products)
        if any(f.native_period >= n for f in facts):
            raise ValueError("no native fact beyond complete closed prefix")
        first = facts[0]
        fares = {o["product_id"]: o["fare"] for o in by_key[key]["menu_view"]["offers"]}
        if _encode(first.fare) != fares[first.product_id]:
            raise ValueError(
                "original purchase fare must match captured SAME-call offer"
            )
        histories[root], tickets_by_root[root] = facts, tickets
        expected_searches[root] = searches
    calls = {root: [] for root in roots}
    previous_call_period = -1
    for row in presentation_calls:
        if (
            type(row) is not NativePresentationRow
            or row.analyst_root not in roots
            or row.analyst_original_key != roots[row.analyst_root]
        ):
            raise ValueError("typed same-source native presentation action required")
        f = row.fact
        if (
            type(f) is not NativePresentationFact
            or f.person_call != len(calls[row.analyst_root])
            or f.person_ticket != 0
            or type(f.airport_counter) is not bool
        ):
            raise ValueError("ALL successful person-local presentation calls required")
        if (
            type(f.native_period) is not int
            or not previous_call_period <= f.native_period <= n
        ):
            raise ValueError(
                "ordered native presentation period including terminal boundary required"
            )
        cancellations = [
            x.native_period
            for x in histories[row.analyst_root]
            if x.kind == "native_ticket_cancelled" and x.person_ticket == 0
        ]
        if not cancellations or min(cancellations) > f.native_period:
            raise ValueError(
                "original native cancellation must precede proof presentation"
            )
        previous_call_period = f.native_period
        calls[row.analyst_root].append(f)
    periods, seen = [], {root: [] for root in roots}
    eligibility = {}
    person_decisions = {key: [] for key in by_key}
    for t in range(n):
        op, pp = own_periods[t], proof_periods[t]
        if (
            type(op) is not OwnInformationPeriod
            or type(pp) is not ProofInformationPeriod
            or op.period != t
            or pp.period != t
            or len(op.reentry_rows) != len(pp.reentry_rows)
        ):
            raise ValueError("complete matching typed own/proof native period required")
        reentry = []
        period_roots = set()
        for o, p in zip(op.reentry_rows, pp.reentry_rows):
            if type(o) is not OwnReentryRow or type(p) is not ProofReentryRow:
                raise TypeError("typed own/proof attempt rows required")
            root, key = o.analyst_root, o.analyst_original_key
            if (
                root not in roots
                or key != roots[root]
                or (p.analyst_root, p.analyst_original_key) != (root, key)
                or root in period_roots
            ):
                raise ValueError(
                    "same unique original root/key across attempt contexts required"
                )
            period_roots.add(root)
            v, q = o.view, p.view
            if type(v) is not OwnReentryView or type(q) is not ProofReentryView:
                raise TypeError("typed native own/proof views required")
            instant = context["clock"][t]
            if (
                v.native_period,
                q.native_period,
                v.compiled_opportunity_utc,
                q.compiled_opportunity_utc,
            ) != (t, t, instant, instant):
                raise ValueError("same native period and compiled opportunity required")
            original = by_key[key]["analyst_person"]
            product = products[tuple(products)[original["native_row"]["chosen"]]]
            if (
                v.original_market != original["request_market"]
                or v.original_carrier != product["carrier"]
            ):
                raise ValueError(
                    "own original market/carrier must match ordinary source"
                )
            first = min(
                f.native_period
                for f in histories[root]
                if f.kind == "native_ticket_cancelled"
            )
            expiry = utc(v.native_exact_expiry_utc).isoformat()
            contract = (
                v.first_stranded_period,
                v.exclusive_deadline_period,
                expiry,
                v.interline_enabled,
            )
            if (
                v.first_stranded_period != first
                or type(v.interline_enabled) is not bool
                or type(v.exclusive_deadline_period) is not int
                or not t < v.exclusive_deadline_period <= len(context["clock"]) - 1
                or utc(instant) >= utc(expiry)
                or v.scope != "native-own-facts-before-current-reentry-marks-only"
                or q.scope
                != "native-presentation-call-history-before-current-reentry-marks-only"
            ):
                raise ValueError(
                    "native own eligibility/first-stranding/expiry/scope differs"
                )
            if root in eligibility and eligibility[root] != contract:
                raise ValueError(
                    "first native deadline/expiry/interline contract cannot change"
                )
            eligibility[root] = contract
            prefix = tuple(v.own_facts)
            full = histories[root]
            if tuple(_encode(jsonable(f)) for f in prefix) != tuple(
                _encode(jsonable(f)) for f in full[: len(prefix)]
            ):
                raise ValueError("earlier own facts must be exact prospective prefix")
            if not prefix or len(prefix) >= len(full):
                raise ValueError(
                    "captured earlier own prefix and current later search fact required"
                )
            current = full[len(prefix)]
            if (
                not current.kind.startswith("native_reentry_")
                or current.native_period != t
            ):
                raise ValueError(
                    "current outcome must stay AFTER pre-marks own context"
                )
            if any(
                f.native_period > t
                or (f.kind.startswith("native_reentry_") and f.native_period >= t)
                for f in prefix
            ):
                raise ValueError(
                    "future/current search outcome cannot enter earlier context"
                )
            if current in seen[root]:
                raise ValueError("native attempt context cannot be duplicated")
            seen[root].append(current)
            earlier = tuple(q.earlier_presentation_calls)
            if (
                tuple(calls[root][: len(earlier)]) != earlier
                or len(earlier) < sum(f.native_period < t for f in calls[root])
                or any(f.native_period > t for f in earlier)
            ):
                raise ValueError("earlier proof calls must be complete captured prefix")
            if (
                type(q.native_proof_on_file) is not bool
                or type(q.native_counter_this_period) is not bool
                or q.native_proof_on_file != bool(earlier)
                or q.native_counter_this_period
                != any(f.native_period == t and f.airport_counter for f in earlier)
            ):
                raise ValueError(
                    "native proof/counter state must match SAME captured earlier actions"
                )
            view = {
                "kind": "native_reentry_before_marks",
                "native_period": t,
                "compiled_opportunity_utc": instant,
                "original_ordinary_menu": by_key[key]["menu_view"],
                "earlier_own": own_reentry_information_view(o),
                "earlier_proof": proof_information_view(p),
                "current_effective_reentry_menu": None,
                "known_private_type": None,
                "received_disclosures_or_complete_knowledge": None,
                "scope": "captured-native-streams-only-no-cross-stream-instant-order",
            }
            ordinal = len(person_decisions[key])
            row = NativeDecisionContext(key, root, ordinal, view)
            person_decisions[key].append((t, "reentry", len(reentry)))
            reentry.append(row)
        admitted = []
        for i, ordinary_row in enumerate(context["periods"][t]["rows"]):
            person = ordinary_row["analyst_person"]
            key = tuple(person["key"])
            if person_decisions[key]:
                raise ValueError(
                    "ordinary original decision must precede later own searches"
                )
            view = {
                "kind": "native_ordinary_before_taste_utility_choice",
                "native_period": t,
                "compiled_opportunity_utc": context["clock"][t],
                "ordinary_menu": ordinary_row["menu_view"],
                "known_private_type": None,
                "received_disclosures_or_complete_knowledge": None,
            }
            admitted.append(
                NativeDecisionContext(key, person["original_record"], 0, view)
            )
            person_decisions[key].append((t, "ordinary", i))
        periods.append(NativeContextPeriod(t, tuple(reentry), tuple(admitted)))
    if any(tuple(seen[root]) != expected_searches[root] for root in roots):
        raise ValueError("EVERY native search needs one own/proof context, no omission")
    people = tuple(
        {
            "analyst_original_key": key,
            "analyst_root": by_key[key]["analyst_person"]["original_record"],
            "context_coordinates": tuple(person_decisions[key]),
        }
        for key in by_key
    )
    analyst = {
        "supplied_source": source,
        "own_periods": _encode(jsonable(own_periods)),
        "proof_periods": _encode(jsonable(proof_periods)),
        "final_prospective_histories": _encode(jsonable(final_histories)),
        "presentation_calls": _encode(jsonable(presentation_calls)),
    }
    result = NativeInformationContexts(
        analyst, tuple(periods), people, definition, row_limit
    )
    if _units(jsonable(result)) > row_limit:
        raise NativeInformationContextLimit(
            "ALL source/fact/proof/people/context output exceeds limit; no partial panel"
        )
    return result


def native_decision_context_view(row):
    if type(row) is not NativeDecisionContext:
        raise TypeError("native decision-context row required")
    return _encode(jsonable(row.view))


def native_information_context_record(panel):
    if type(panel) is not NativeInformationContexts or panel.rule != RULE:
        raise TypeError("complete native information-context panel required")
    return {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "result": _encode(jsonable(panel)),
        "actual_received_knowledge_type_or_native_probability_authenticated": False,
        "final_reentry_menu_utility_CS_service_or_payoff": None,
        "native_behavior_actors_observations_metrics_or_defaults_changed": False,
        "scientific_budget": None,
    }
