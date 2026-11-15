"""A.10/A.5/A.14/B0/V9: pure original-choice/closing-boundary association.

Complete supplied native records are structural evidence, not authenticated
probabilities, received knowledge, preferences or lifecycle welfare.
"""

from dataclasses import dataclass

from airline_market.core.hashing import canonical_json
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.booking_loss_law import (
    _limit,
    _validate,
    booking_loss_history,
    booking_loss_record,
    compile_booking_loss_law,
)
from airline_market.sim.causal_information import (
    DecisionInput,
    OwnAction,
    _require,
    causal_decision_view,
)
from airline_market.sim.native_causal_information import (
    NativeCausalInput,
    NativeCausalPeriod,
)
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _complete_source, _encoded, _owned

RULE = "complete-native-original-choice-boundary-binding-v1"


class ChoiceBoundaryLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class ChoiceBoundaryAssociation:
    original_person_label: str
    definition: str
    source_ref: str


@dataclass(frozen=True)
class ChoiceBoundaryBinding:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", deep_freeze(self.analyst))
        object.__setattr__(self, "view", deep_freeze(self.view))


def _same(a, b):
    return canonical_json(_encoded(a)) == canonical_json(_encoded(b))


def choice_boundary_conditioning_context(ordinary_input):
    """Full supplied earlier view and separate analyst computation menu only."""
    if (
        type(ordinary_input) is not NativeCausalInput
        or ordinary_input.kind != "ordinary"
    ):
        raise TypeError("typed genuine ordinary input required")
    view = causal_decision_view(ordinary_input.receipt)
    return deep_freeze(
        {
            "prechoice_person_view": _encoded(view),
            "same_call_computation_menu": _encoded(ordinary_input.computation_input),
            "scope": "full supplied I0; not authenticated complete passenger knowledge",
        }
    )


def _source(source, periods, selected):
    if (
        source["failed"]
        or source["pending"] is not None
        or source["native_parent_steps_completed"] != len(periods)
        or not periods
    ):
        raise ValueError("complete nonempty successful native prefix required")
    _complete_source(source, selected)
    if not _same(source["periods"], periods):
        raise ValueError("ALL supplied typed native periods and inputs required")
    closed = periods[-1].journal
    if not _same(source["journal"]["analyst_context"], closed):
        raise ValueError(
            "exact native closing journal required; no post-close tail filtering"
        )
    state = _require(closed)
    if (
        state["pending"] is not None
        or source["journal"]["pending_declared_decision"] is not None
    ):
        raise ValueError("complete actual input/result support required")
    decisions = []
    for i, period in enumerate(periods):
        if period.period != i or period.native_world_t != i + 1:
            raise ValueError("consecutive full native closures required")
        if _require(period.journal)["pending"] is not None:
            raise ValueError("no incomplete period journal")
        for event in period.journal.events:
            closing = event.position.anchor_utc == closed.clock[i + 1]
            if event.position.anchor_utc > closed.clock[i + 1] or (
                closing
                and not (
                    type(event.payload) is OwnAction
                    and event.payload.kind == "cancellation"
                )
            ):
                raise ValueError(
                    "earlier native close cannot include future registration/input/result/message actions"
                )
        for row in period.inputs:
            if type(row) is not NativeCausalInput:
                raise TypeError("all typed actual input rows required")
            causal_decision_view(row.receipt)
            parent = source["parent_source"]
            parent_menu = (
                parent["source_context"]["ordinary_information"][i]["rows"][
                    row.parent_input_index
                ]["menu_view"]
                if row.kind == "ordinary"
                else parent["periods"][i]["inputs"][row.parent_input_index]["view"]
            )
            if not _same(row.computation_input, parent_menu):
                raise ValueError("same actual parent computation menu required")
            prefix = row.receipt.source_journal.events
            event = prefix[-1]
            if (
                type(event.payload) is not DecisionInput
                or event.payload.stage != row.kind
                or event.person_key != row.analyst_original_key
                or event.position.anchor_utc != closed.clock[i]
                or not _same(prefix, period.journal.events[: len(prefix)])
            ):
                raise ValueError(
                    "same exact immutable input-ending global prefix required"
                )
            if event.payload.decision_id not in state["results"]:
                raise ValueError(
                    "every actual input requires its complete native result"
                )
            decisions.append(event.payload.decision_id)
    if len(set(decisions)) != len(decisions) or set(decisions) != set(
        state["decisions"]
    ):
        raise ValueError("ALL and ONLY native decision inputs required")
    if selected.analyst_root is not None or not any(
        _same(selected, row) for row in periods[-1].inputs
    ):
        raise ValueError("original ordinary input from LAST closed period required")
    return closed, state


def _people(context, state):
    products = context["native_spec"]["products"]
    original = {}
    all_keys = []
    for period in context["ordinary_information"]:
        for row in period["rows"]:
            person = row["analyst_person"]
            key = tuple(person["key"])
            all_keys.append(key)
            root = person["original_record"]
            native_row = person["native_row"]
            if (
                person["request_market"]
                != context["native_spec"]["markets"][key[2]]["id"]
                or native_row["period"] != key[0]
                or native_row["market"] != key[2]
            ):
                raise ValueError(
                    "same native original period/market coordinates required"
                )
            expected_catalogue = tuple(
                p["id"] for p in products if p["market"] == person["request_market"]
            )
            if (
                tuple(o["product_id"] for o in row["menu_view"]["offers"])
                != expected_catalogue
            ):
                raise ValueError("complete native market product catalogue required")
            decision = next(
                e
                for e in state["decisions"].values()
                if e.person_key == key and e.payload.stage == "ordinary"
            )
            actual = state["results"][decision.payload.decision_id].payload
            chosen = person["native_row"]["chosen"]
            expected = None if chosen is None else products[chosen]["id"]
            if actual.product_id != expected or (root is None) != actual.outside:
                raise ValueError("complete original choice/outside support differs")
            if row["menu_view"]["market_id"] != person["request_market"]:
                raise ValueError("same original request market required")
            if root is not None:
                if type(root) is not int or root < 0 or str(root) in original:
                    raise ValueError("unique native original roots required")
                original[str(root)] = (person, row["menu_view"], expected)
    if len(set(all_keys)) != len(all_keys) or set(all_keys) != set(state["people"]):
        raise ValueError("all original people including outside required")
    for field in ("own_keys", "own_tickets", "final_own_histories"):
        if set(context[field]) != set(original):
            raise ValueError("full original booked-root partition required")
    tickets = []
    for root, (person, menu, product_id) in original.items():
        owned = context["own_tickets"][root]
        facts = context["final_own_histories"][root]
        if (
            not owned
            or owned[0] != int(root)
            or any(type(t) is not int or t < 0 for t in owned)
            or not facts
            or tuple(context["own_keys"][root]) != tuple(person["key"])
        ):
            raise ValueError("prospective root/key/ticket histories required")
        tickets.extend(owned)
        first = facts[0]
        offer = next(o for o in menu["offers"] if o["product_id"] == product_id)
        if (
            first["ordinal"] != 0
            or first["kind"] != "native_ordinary_purchase"
            or first["native_period"] != person["key"][0]
            or first["stage"] is not None
            or first["person_ticket"] != 0
            or first["product_id"] != product_id
            or not offer["offered"]
            or not _same(first["fare"], offer["fare"])
        ):
            raise ValueError(
                "actual original purchase/product/offered binary64 fare required"
            )
    if len(tickets) != len(set(tickets)):
        raise ValueError("no ticket transfer or duplicate ticket across roots")
    return original


def bind_choice_boundary(
    law, ordinary_input, native_periods, source_record, association, *, row_limit
):
    """Compile ONE NEW declared law after complete last-period source association.

    X remains latent; observe only the actual original product/outside and
    cancellation Boolean. Richer selected-person tails require another model.
    """
    shape = _validate(law)
    if (
        type(ordinary_input) is not NativeCausalInput
        or ordinary_input.kind != "ordinary"
    ):
        raise TypeError("typed actual ordinary input required")
    periods = tuple(native_periods)
    if any(type(p) is not NativeCausalPeriod for p in periods):
        raise TypeError("full typed native causal periods required")
    if type(association) is not ChoiceBoundaryAssociation:
        raise TypeError("explicit original logical-label association required")
    for name in association.__dataclass_fields__:
        _text(getattr(association, name))
    if association.original_person_label != law.original_person_label:
        raise ValueError("same explicit logical-person label required")
    source = _owned(source_record)
    backend_bound = _limit(law, shape, row_limit)
    inputs = _owned(
        _encoded(
            {
                "source": source,
                "periods": periods,
                "ordinary": ordinary_input,
                "association": association,
            }
        )
    )
    bound = 4 * _units(_encoded(inputs)) + 4 * backend_bound + 10000
    if bound > row_limit:
        raise ChoiceBoundaryLimit(
            "ALL source/input/zero-law/output support exceeds limit; no partial binding"
        )
    closed, state = _source(source, periods, ordinary_input)
    context = source["parent_source"]["source_context"]
    original = _people(context, state)
    prechoice = choice_boundary_conditioning_context(ordinary_input)
    if not _same(law.conditioning_context, prechoice):
        raise ValueError(
            "law must condition on full saved BEFORE-choice view and separate menu"
        )
    menu = ordinary_input.computation_input
    if not _same(law.products, menu["offers"]):
        raise ValueError(
            "same complete actual computation catalogue/fares/offers required"
        )
    selected_event = ordinary_input.receipt.source_journal.events[-1]
    key = ordinary_input.analyst_original_key
    result = state["results"][selected_event.payload.decision_id]
    product_id = result.payload.product_id
    n = len(periods)
    tail = [
        e
        for e in closed.events[len(ordinary_input.receipt.source_journal.events) :]
        if e.person_key == key
    ]
    if (
        not tail
        or tail[0] != result
        or any(
            type(e.payload) is not OwnAction or e.payload.kind != "cancellation"
            for e in tail[1:]
        )
    ):
        raise ValueError(
            "only actual choice and optional cancellation in selected-person tail supported"
        )
    if len(tail) > 2:
        raise ValueError(
            "no repeated cancellation or additional selected-person history"
        )
    root = next(
        (r for r, (person, _, _) in original.items() if tuple(person["key"]) == key),
        None,
    )
    cancelled = len(tail) == 2
    if product_id is None:
        if root is not None or cancelled:
            raise ValueError("outside original person has no ticket/cancellation")
    else:
        if root is None or tuple(context["own_tickets"][root]) != (int(root),):
            raise ValueError(
                "same original ticket only; replacements unsupported for selected new person"
            )
        facts = context["final_own_histories"][root]
        if len(facts) != 1 + int(cancelled):
            raise ValueError("ALL prospective selected-person facts required")
        if cancelled:
            fact = facts[1]
            if (
                tail[1].position.anchor_utc
                not in (closed.clock[n - 1], closed.clock[n])
                or fact["native_period"]
                != closed.clock.index(tail[1].position.anchor_utc)
                or fact["ordinal"] != 1
                or fact["kind"] != "native_ticket_cancelled"
                or fact["stage"] is not None
                or fact["person_ticket"] != 0
                or fact["product_id"] != product_id
                or not _same(fact["fare"], facts[0]["fare"])
            ):
                raise ValueError(
                    "same actual original-ticket cancellation and unchanged closing-n anchor required"
                )
    compiled = compile_booking_loss_law(law, row_limit=row_limit)
    history = booking_loss_history(compiled, product_id, cancelled=cancelled)
    analyst = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "inputs": inputs,
        "law_result": booking_loss_record(compiled),
        "selected_history": history,
        "actual_original_key": key,
        "actual_root": None if root is None else int(root),
        "actual_product": product_id,
        "cancelled": cancelled,
        "closed_native_periods": n,
        "structural_bound": bound,
        "native_probability_or_knowledge_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return ChoiceBoundaryBinding(analyst, prechoice, row_limit)


def choice_boundary_record(binding):
    if type(binding) is not ChoiceBoundaryBinding or binding.rule != RULE:
        raise TypeError("complete typed choice/boundary binding required")
    return _encoded(binding)


def choice_boundary_prechoice_view(binding):
    if type(binding) is not ChoiceBoundaryBinding or binding.rule != RULE:
        raise TypeError("complete typed choice/boundary binding required")
    return _encoded(binding.view["prechoice_person_view"])
