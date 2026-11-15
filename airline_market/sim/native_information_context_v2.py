"""A.10/A.5/A.14/B0/V9: native context market/stage consistency revision.

NEW pure join revision on NEW inputs checks actual declared
purchase product/market/own-interline stage and ordinary offered-choice support.
No source repair, booking feasibility inference, native behavior or utility change.
"""

from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim import native_information_context as parent
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.received_information import _units

RULE = "complete-native-contexts-market-stage-consistency-v2"


@dataclass(frozen=True)
class NativeInformationContextsV2:
    native: parent.NativeInformationContexts
    purchase_support: tuple
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "purchase_support", deep_freeze(self.purchase_support))


def _purchase_support(panel):
    context = panel.analyst_context["supplied_source"]["analyst_context"]
    products = context["native_spec"]["products"]
    by_id = {p["id"]: p for p in products}
    people = tuple(r for p in context["periods"] for r in p["rows"])
    roots = {}
    audit = []
    for row in people:
        person, menu = row["analyst_person"], row["menu_view"]
        key, root = tuple(person["key"]), person["original_record"]
        chosen = person["native_row"]["chosen"]
        if (chosen is None) != (root is None):
            raise ValueError(
                "ordinary selected product and original ticket partition differs"
            )
        item = {
            "analyst_original_key": key,
            "analyst_root": root,
            "ordinary_selected_product": None,
            "ordinary_selected_offer_enabled": None,
            "ticket_purchases": (),
        }
        if chosen is None:
            if menu["outside_available"] is not True:
                raise ValueError(
                    "actual ordinary outside choice requires declared offered outside"
                )
        else:
            if type(chosen) is not int or not 0 <= chosen < len(products):
                raise ValueError(
                    "actual named ordinary physical product column required"
                )
            product = products[chosen]
            if (
                product["market"] != person["request_market"]
                or menu["market_id"] != person["request_market"]
            ):
                raise ValueError(
                    "ordinary physical choice must serve original request market"
                )
            offers = tuple(
                o for o in menu["offers"] if o["product_id"] == product["id"]
            )
            if len(offers) != 1 or offers[0]["offered"] is not True:
                raise ValueError(
                    "ordinary chosen product requires SAME captured offered flag"
                )
            item.update(
                ordinary_selected_product=product["id"],
                ordinary_selected_offer_enabled=True,
            )
            roots[root] = (person, product["carrier"], item)
        audit.append(item)
    attempts = {
        (r.analyst_root, p.period): r for p in panel.periods for r in p.reentry_contexts
    }
    for root, facts, tickets, key in panel.analyst_context[
        "final_prospective_histories"
    ]:
        person, original_carrier, item = roots[root]
        checks = []
        for fact in facts:
            if fact["kind"] not in {
                "native_ordinary_purchase",
                "native_reentry_purchase",
            }:
                continue
            product = by_id[fact["product_id"]]
            if product["market"] != person["request_market"]:
                raise ValueError(
                    "EVERY original/replacement product must serve SAME original market"
                )
            stage = fact["stage"]
            if fact["kind"] == "native_reentry_purchase":
                view = attempts[root, fact["native_period"]].view["earlier_own"]
                same = product["carrier"] == original_carrier
                if (stage == "own" and not same) or (stage == "interline" and same):
                    raise ValueError(
                        "actual product carrier and recorded own/interline stage differ"
                    )
                if stage == "interline" and view["interline_enabled"] is not True:
                    raise ValueError(
                        "recorded interline purchase requires SAME native interline contract"
                    )
            checks.append(
                {
                    "person_ticket": fact["person_ticket"],
                    "product_id": product["id"],
                    "request_market": person["request_market"],
                    "product_market": product["market"],
                    "original_carrier": original_carrier,
                    "product_carrier": product["carrier"],
                    "stage": stage,
                    "fare_metadata": fact["fare"],
                }
            )
        if len(checks) != len(tickets):
            raise ValueError(
                "ALL declared native purchase tickets require market/stage support"
            )
        item["ticket_purchases"] = tuple(checks)
    return tuple(audit)


def _join_current_prefix(
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
    parent._text(definition)
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive all-support infrastructure limit required")
    source = parent.received_information_record(journal)
    context = source["analyst_context"]
    n = context["completed_source_periods"]
    (own_periods, proof_periods) = (tuple(own_periods), tuple(proof_periods))
    (final_histories, presentation_calls) = (
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
    (histories, tickets_by_root, expected_searches) = ({}, {}, {})
    all_tickets = set()
    if len(final_histories) != len(roots):
        raise ValueError(
            "EVERY originally booked root final prospective history required"
        )
    for row in final_histories:
        if len(row) != 4:
            raise ValueError("complete root/facts/tickets/original-key row required")
        (root, facts, tickets, key) = row
        (key, facts, tickets) = (parent._key(key), tuple(facts), tuple(tickets))
        if (
            type(root) is not int
            or root not in roots
            or roots[root] != key
            or (root in histories)
        ):
            raise ValueError("same source original-person root association required")
        if any(type(t) is not int or t < 0 or t in all_tickets for t in tickets) or len(
            set(tickets)
        ) != len(tickets):
            raise ValueError("global tickets cannot transfer between original people")
        all_tickets.update(tickets)
        searches = parent._facts(
            facts, tickets, by_key[key]["analyst_person"], products
        )
        if any(
            f.native_period > n
            or (f.native_period == n and f.kind != "native_ticket_cancelled")
            for f in facts
        ):
            raise ValueError(
                "only cancellation facts may lie at the declared closing boundary"
            )
        first = facts[0]
        fares = {o["product_id"]: o["fare"] for o in by_key[key]["menu_view"]["offers"]}
        if _encode(first.fare) != fares[first.product_id]:
            raise ValueError(
                "original purchase fare must match captured SAME-call offer"
            )
        (histories[root], tickets_by_root[root]) = (facts, tickets)
        expected_searches[root] = searches
    calls = {root: [] for root in roots}
    previous_call_period = -1
    for row in presentation_calls:
        if (
            type(row) is not parent.NativePresentationRow
            or row.analyst_root not in roots
            or row.analyst_original_key != roots[row.analyst_root]
        ):
            raise ValueError("typed same-source native presentation action required")
        f = row.fact
        if (
            type(f) is not parent.NativePresentationFact
            or f.person_call != len(calls[row.analyst_root])
            or f.person_ticket != 0
            or (type(f.airport_counter) is not bool)
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
    (periods, seen) = ([], {root: [] for root in roots})
    eligibility = {}
    person_decisions = {key: [] for key in by_key}
    for t in range(n):
        (op, pp) = (own_periods[t], proof_periods[t])
        if (
            type(op) is not parent.OwnInformationPeriod
            or type(pp) is not parent.ProofInformationPeriod
            or op.period != t
            or (pp.period != t)
            or (len(op.reentry_rows) != len(pp.reentry_rows))
        ):
            raise ValueError("complete matching typed own/proof native period required")
        reentry = []
        period_roots = set()
        for o, p in zip(op.reentry_rows, pp.reentry_rows):
            if (
                type(o) is not parent.OwnReentryRow
                or type(p) is not parent.ProofReentryRow
            ):
                raise TypeError("typed own/proof attempt rows required")
            (root, key) = (o.analyst_root, o.analyst_original_key)
            if (
                root not in roots
                or key != roots[root]
                or (p.analyst_root, p.analyst_original_key) != (root, key)
                or (root in period_roots)
            ):
                raise ValueError(
                    "same unique original root/key across attempt contexts required"
                )
            period_roots.add(root)
            (v, q) = (o.view, p.view)
            if (
                type(v) is not parent.OwnReentryView
                or type(q) is not parent.ProofReentryView
            ):
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
            expiry = parent.utc(v.native_exact_expiry_utc).isoformat()
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
                or (not t < v.exclusive_deadline_period <= len(context["clock"]) - 1)
                or (parent.utc(instant) >= parent.utc(expiry))
                or (v.scope != "native-own-facts-before-current-reentry-marks-only")
                or (
                    q.scope
                    != "native-presentation-call-history-before-current-reentry-marks-only"
                )
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
                or (
                    q.native_counter_this_period
                    != any(f.native_period == t and f.airport_counter for f in earlier)
                )
            ):
                raise ValueError(
                    "native proof/counter state must match SAME captured earlier actions"
                )
            view = {
                "kind": "native_reentry_before_marks",
                "native_period": t,
                "compiled_opportunity_utc": instant,
                "original_ordinary_menu": by_key[key]["menu_view"],
                "earlier_own": parent.own_reentry_information_view(o),
                "earlier_proof": parent.proof_information_view(p),
                "current_effective_reentry_menu": None,
                "known_private_type": None,
                "received_disclosures_or_complete_knowledge": None,
                "scope": "captured-native-streams-only-no-cross-stream-instant-order",
            }
            ordinal = len(person_decisions[key])
            row = parent.NativeDecisionContext(key, root, ordinal, view)
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
                parent.NativeDecisionContext(key, person["original_record"], 0, view)
            )
            person_decisions[key].append((t, "ordinary", i))
        periods.append(parent.NativeContextPeriod(t, tuple(reentry), tuple(admitted)))
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
    result = parent.NativeInformationContexts(
        analyst, tuple(periods), people, definition, row_limit
    )
    if _units(jsonable(result)) > row_limit:
        raise parent.NativeInformationContextLimit(
            "ALL source/fact/proof/people/context output exceeds limit; no partial panel"
        )
    return result


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
    """One NEW pure join, then full same-O&D/stage/offered-choice checks.

    Old v1 stays LIMITED for these gaps and closing-boundary cancellations. This does
    not validate final rescue/reserve stage masks/fares or actual public truth.
    """
    native = _join_current_prefix(
        journal,
        own_periods,
        proof_periods,
        final_histories,
        presentation_calls,
        definition=definition,
        row_limit=row_limit,
    )
    checks = _purchase_support(native)
    result = NativeInformationContextsV2(native, checks, row_limit)
    if _units(native_information_context_record(result)) > row_limit:
        raise parent.NativeInformationContextLimit(
            "ALL parent support + NEW market/stage audit and wrapper metadata exceed limit; no partial result"
        )
    return result


def native_information_context_record(panel):
    """Owned full NEW supplied-source export; no producer/solver/choice call."""
    if type(panel) is not NativeInformationContextsV2 or panel.rule != RULE:
        raise TypeError("complete native-context v2 result required")
    return {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "result": _encode(jsonable(panel)),
        "ordinary_choice_same_offer_and_purchase_market_carrier_stage_checked": True,
        "final_reentry_menu_actual_knowledge_type_probability_or_utility_authenticated": False,
        "native_behavior_actors_observations_metrics_or_defaults_changed": False,
        "lifecycle_utility_CS_service_or_payoff": None,
        "scientific_budget": None,
    }
