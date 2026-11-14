"""A.8/B0/V9: static named-priority rationing and exact assignment bound.

One fixed posting opportunity. Applicant utilities, objective scores, eligibility
and capacity net of existing bookings/holds are supplied explicitly. No native
booking/price changes, empirical inference or old planner adoption occurs here.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from numbers import Integral

from airline_market.core.rng import StreamRegistry


def rational(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is bool:
        raise ValueError("finite declared score required")
    try:
        x = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("finite declared decimal score required") from error
    if not x.is_finite():
        raise ValueError("finite declared score required")
    return Fraction(x)


@dataclass(frozen=True)
class Offer:
    product: str
    choice_value: Fraction
    allocation_value: Fraction

    def __post_init__(self):
        if not isinstance(self.product, str) or not self.product:
            raise ValueError("explicit physical product ID required")
        object.__setattr__(self, "choice_value", rational(self.choice_value))
        object.__setattr__(self, "allocation_value", rational(self.allocation_value))


@dataclass(frozen=True)
class Applicant:
    key: tuple
    offers: tuple[Offer, ...]

    def __post_init__(self):
        key = tuple(self.key)
        offers = tuple(self.offers)
        if len(key) != 3 or any(type(x) is not int or x < 0 for x in key):
            raise ValueError(
                "nonnegative original (period,rank,market) coordinates required"
            )
        if any(type(x) is not Offer for x in offers) or len(
            {x.product for x in offers}
        ) != len(offers):
            raise ValueError("one offer per physical product/taste required")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "offers", offers)


@dataclass(frozen=True)
class AllocationProblem:
    period: int
    capacities: tuple
    products: tuple
    applicants: tuple[Applicant, ...]

    def __post_init__(self):
        stock = tuple(sorted(tuple(x) for x in self.capacities))
        products = tuple(sorted((p, tuple(legs)) for p, legs in self.products))
        raw_people = tuple(self.applicants)
        if any(type(x) is not Applicant for x in raw_people):
            raise ValueError("explicit Applicant records required")
        people = tuple(sorted(raw_people, key=lambda x: x.key))
        if type(self.period) is not int or self.period < 0:
            raise ValueError("explicit native period required")
        if len(dict(stock)) != len(stock) or any(
            not isinstance(k, str)
            or not k
            or not isinstance(n, Integral)
            or type(n) is bool
            or n < 0
            for k, n in stock
        ):
            raise ValueError(
                "distinct physical legs and nonnegative integer free capacity required"
            )
        if len(dict(products)) != len(products) or any(
            not isinstance(p, str)
            or not p
            or not legs
            or len(set(legs)) != len(legs)
            or set(legs) - set(dict(stock))
            for p, legs in products
        ):
            raise ValueError(
                "distinct physical products with known distinct itinerary legs required"
            )
        if (
            any(type(x) is not Applicant for x in people)
            or len({x.key[:2] for x in people}) != len(people)
            or any(x.key[0] != self.period for x in people)
        ):
            raise ValueError(
                "one original request per rank in this posting opportunity required"
            )
        if any(o.product not in dict(products) for x in people for o in x.offers):
            raise ValueError("unknown physical offer product")
        object.__setattr__(self, "capacities", tuple((k, int(n)) for k, n in stock))
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "applicants", people)


def from_compiled(compiled, period, free_capacity, applicants):
    """Use fixed native itinerary incidence; caller supplies net free leg stock."""
    stock = dict(free_capacity)
    if set(stock) != set(compiled.leg_index):
        raise ValueError("free capacity must explicitly cover every compiled leg")
    return AllocationProblem(
        period,
        tuple(stock.items()),
        tuple((p.id, p.legs) for p in compiled.spec.products),
        tuple(applicants),
    )


def _layout(problem):
    if type(problem) is not AllocationProblem:
        raise TypeError("AllocationProblem required")
    legs = tuple(k for k, n in problem.capacities)
    initial = tuple(n for k, n in problem.capacities)
    incidence = {p: tuple(legs.index(k) for k in path) for p, path in problem.products}
    return legs, initial, incidence


def _consume(stock, use):
    if any(stock[k] == 0 for k in use):
        return None
    new = list(stock)
    for k in use:
        new[k] -= 1
    return tuple(new)


def _result(problem, legs, initial, remaining, assignments, mode, value, **extra):
    return {
        "plan_id": "A.8/B0/V9",
        "mode": mode,
        "assignments": tuple(assignments),
        "allocation_value": value,
        "capacity": tuple(
            {"leg": k, "initial": a, "used": a - b, "remaining": b}
            for k, a, b in zip(legs, initial, remaining)
        ),
        "unassigned_requests": sum(p is None for key, p in assignments),
        "supplied_requests": len(problem.applicants),
        "native_engine_adopted": False,
        "dynamic_or_equilibrium_bound": False,
        **extra,
    }


def priority_allocation(problem, *, binding_cap, registry=None):
    """Uniform permutation of the full supplied pool only when explicitly binding.

    Same physical pool/seed/period has the same lottery regardless of input order
    or endogenous booking IDs. Keep ALL physical applicants across cap contrasts.
    Without a binding cap, original chronological ranks and native offer ties are
    retained and no random stream is touched. Outside is last with value zero.
    """
    legs, initial, incidence = _layout(problem)
    if type(binding_cap) is not bool:
        raise TypeError("explicit binding-cap boolean required")
    people = problem.applicants
    namespace = None
    if binding_cap:
        if not isinstance(registry, StreamRegistry):
            raise TypeError("named StreamRegistry required for lottery")
        namespace = registry.namespace + "/A8_priority_v1"
        stream = StreamRegistry(registry.root_seed, namespace).generator(problem.period)
        order = tuple(int(i) for i in stream.permutation(len(people)))
    else:
        order = tuple(range(len(people)))
    stock = initial
    assignments = []
    value = Fraction(0)
    for index in order:
        person = people[index]
        best = None
        best_value = Fraction(0)
        next_stock = stock
        for offer in person.offers:
            remaining = _consume(stock, incidence[offer.product])
            # Product alternatives precede outside in native argmax ties.
            if remaining is not None and (
                offer.choice_value > best_value
                or (best is None and offer.choice_value == 0)
            ):
                best = offer
                best_value = offer.choice_value
                next_stock = remaining
        stock = next_stock
        assignments.append((person.key, None if best is None else best.product))
        if best is not None:
            value += best.allocation_value
    return _result(
        problem,
        legs,
        initial,
        stock,
        assignments,
        "random_priority" if binding_cap else "chronological",
        value,
        priority_namespace=namespace,
        priority_order=tuple(people[i].key for i in order),
        binding_cap_supplied=binding_cap,
    )


class AllocationLimitError(RuntimeError):
    """Caller infrastructure cap reached; no partial bound/value is returned."""


def exact_assignment_bound(problem, *, state_limit):
    """Maximize declared allocation score over weakly willing assignments.

    This is a static individual-rational assignment relaxation at fixed prices;
    it can ration alternatives and is not a truthful mechanism/equilibrium bound.
    Inputs use exact rational declared scores. Outside-first ties are fixed before
    outcomes. state_limit is caller infrastructure accounting, not an F9 budget.
    """
    legs, initial, incidence = _layout(problem)
    if type(state_limit) is not int or state_limit < 1:
        raise ValueError("explicit positive state limit required")
    visits = 0

    @lru_cache(None)
    def solve(index, stock):
        nonlocal visits
        visits += 1
        if visits > state_limit:
            raise AllocationLimitError(
                "static assignment state limit; no partial bound"
            )
        if index == len(problem.applicants):
            return Fraction(0), (), stock
        future, choices, remaining = solve(index + 1, stock)
        best = (future, (None,) + choices, remaining)
        for offer in problem.applicants[index].offers:
            if offer.choice_value < 0:
                continue
            new = _consume(stock, incidence[offer.product])
            if new is None:
                continue
            future, choices, remaining = solve(index + 1, new)
            score = offer.allocation_value + future
            if score > best[0]:
                best = (score, (offer.product,) + choices, remaining)
        return best

    value, choices, stock = solve(0, initial)
    assignments = tuple(
        (person.key, p) for person, p in zip(problem.applicants, choices)
    )
    return _result(
        problem,
        legs,
        initial,
        stock,
        assignments,
        "exact_assignment_relaxation",
        value,
        visited_states=visits,
        caller_state_limit=state_limit,
        individual_rationality="choice_value >= outside value zero",
        tie_rule="outside-first, then supplied native offer order",
    )


def result_record(result):
    """Exact rational score envelope for JSON receipts; no numeric rounding."""
    value = result["allocation_value"]
    if type(value) is not Fraction:
        raise TypeError("native allocation result with exact Fraction required")
    return {
        **result,
        "allocation_value": {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        },
    }
