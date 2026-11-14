"""Specification consistency checks with JSON-path errors."""

import math
from ..core.errors import SpecError
from .model import MarketSpec
from .schedule import PeriodSchedule


def _require(condition: bool, path: str, message: str) -> None:
    """
    Raise a path-qualified specification error when an invariant fails (Eqs. 1–19).
    """
    if not condition:
        raise SpecError(path, message)


def _prob(values, path: str, size: int | None = None) -> None:
    """Validate a finite categorical probability vector (Eqs. 1–19)."""
    _require(
        bool(values) and (size is None or len(values) == size),
        path,
        "probability vector has incorrect length",
    )
    _require(
        all(math.isfinite(float(x)) and x >= 0 for x in values),
        path,
        "probabilities must be finite and nonnegative",
    )
    _require(abs(sum(values) - 1) <= 1e-12, path, "probabilities must sum to one")


def validate_spec(s: MarketSpec) -> None:
    """
    Check network, distributions, schedules, pricing and information references (Eqs.
    1–19).
    """
    _require(
        s.schema_version == "airline-market-spec/1",
        "$.schema_version",
        "unsupported schema version",
    )
    T = s.horizon.n_periods
    _require(type(T) is int and T >= 2, "$.horizon.n_periods", "must be an integer >=2")
    if s.horizon.period_days is not None:
        _require(
            len(s.horizon.period_days) == T
            and all(x > 0 for x in s.horizon.period_days),
            "$.horizon.period_days",
            "requires T positive durations",
        )
    indexes = {}
    for name, attr in [
        ("airports", "code"),
        ("carriers", "code"),
        ("legs", "id"),
        ("products", "id"),
        ("markets", "id"),
        ("segments", "id"),
    ]:
        items = getattr(s, name)
        ids = [getattr(x, attr) for x in items]
        _require(
            bool(ids) and len(ids) == len(set(ids)),
            f"$.{name}",
            "IDs must be nonempty and unique",
        )
        indexes[name] = {getattr(x, attr): x for x in items}

    def capacity(c, path):
        """
        Validate a nonnegative integer capacity support and its probability law (Eqs.
        1–19).
        """
        _prob(c.probabilities, path + ".probabilities", len(c.support))
        _require(
            all(type(x) is int and x >= 0 for x in c.support),
            path + ".support",
            "requires nonnegative integer capacities",
        )
        _require(
            len(set(c.support)) == len(c.support),
            path + ".support",
            "duplicate support values",
        )

    for i, l in enumerate(s.legs):
        p = f"$.legs[{i}]"
        _require(l.carrier in indexes["carriers"], p + ".carrier", "unknown carrier")
        _require(
            l.origin in indexes["airports"] and l.dest in indexes["airports"],
            p,
            "unknown airport",
        )
        capacity(l.capacity, p + ".capacity")
    used = set()
    for i, prior in enumerate(s.capacity_priors):
        p = f"$.capacity_priors[{i}]"
        _require(
            bool(prior.legs) and all(l in indexes["legs"] for l in prior.legs),
            p + ".legs",
            "unknown/empty legs",
        )
        _require(
            len(set(indexes["legs"][l].carrier for l in prior.legs)) == 1,
            p + ".legs",
            "joint prior crosses carriers",
        )
        _require(
            not used.intersection(prior.legs)
            and len(set(prior.legs)) == len(prior.legs),
            p + ".legs",
            "overlapping joint priors",
        )
        used.update(prior.legs)
        _prob(prior.probabilities, p + ".probabilities", len(prior.support))
        _require(
            all(
                len(v) == len(prior.legs) and all(type(x) is int and x >= 0 for x in v)
                for v in prior.support
            ),
            p + ".support",
            "invalid capacity vectors",
        )
    for i, m in enumerate(s.markets):
        _require(
            m.origin in indexes["airports"] and m.dest in indexes["airports"],
            f"$.markets[{i}]",
            "unknown market airport",
        )
    for i, p in enumerate(s.products):
        path = f"$.products[{i}]"
        _require(p.carrier in indexes["carriers"], path + ".carrier", "unknown carrier")
        _require(p.market in indexes["markets"], path + ".market", "unknown market")
        _require(
            bool(p.legs) and all(l in indexes["legs"] for l in p.legs),
            path + ".legs",
            "unknown/empty legs",
        )
        legs = [indexes["legs"][l] for l in p.legs]
        _require(
            all(l.carrier == p.carrier for l in legs),
            path + ".legs",
            "legs must belong to product carrier",
        )
        _require(len(set(p.legs)) == len(p.legs), path + ".legs", "repeated dated leg")
        _require(
            all(a.dest == b.origin for a, b in zip(legs, legs[1:])),
            path + ".legs",
            "legs are not contiguous",
        )
        m = indexes["markets"][p.market]
        _require(
            legs[0].origin == m.origin and legs[-1].dest == m.dest,
            path + ".market",
            "itinerary endpoints differ from market",
        )
        _require(
            all(
                math.isfinite(x) and x > 0
                for x in (p.fare_floor, p.reference_fare, p.fare_cap)
            )
            and p.fare_floor <= p.reference_fare <= p.fare_cap,
            path + ".reference_fare",
            "invalid fare range",
        )
        _require(
            math.isfinite(p.service_cost) and p.service_cost >= 0,
            path + ".service_cost",
            "invalid service cost",
        )

    def schedule(sc, path, size=None, probability=False):
        """Reshape and bound the per-block coefficient candidate (Eqs. 1–19)."""
        _require(isinstance(sc, PeriodSchedule), path, "schedule required")
        _require(
            all(type(t) is int and 0 <= t < T for t in sc.breakpoints),
            path + ".breakpoints",
            "breakpoint outside horizon",
        )
        shape = None
        for value in sc.values:
            if size is not None:
                _prob(value, path, size)
            elif probability:
                _require(
                    math.isfinite(value) and 0 <= value <= 1,
                    path,
                    "probability outside [0,1]",
                )
            else:
                _require(
                    math.isfinite(value) and value >= 0,
                    path,
                    "negative/nonfinite intensity",
                )

    d = s.demand
    _require(
        d.mode in ("single_request_bernoulli", "poisson"),
        "$.demand.mode",
        "unknown arrival mode",
    )
    if d.mode == "single_request_bernoulli":
        schedule(
            d.arrival_probability, "$.demand.arrival_probability", probability=True
        )
        schedule(d.market_weights, "$.demand.market_weights", len(s.markets))
    else:
        _require(
            d.intensity is not None, "$.demand.intensity", "Poisson intensity required"
        )
        _require(
            set(d.intensity) == set(indexes["markets"]),
            "$.demand.intensity",
            "requires an intensity per market",
        )
        for m, sc in d.intensity.items():
            schedule(sc, "$.demand.intensity." + m)
    if isinstance(d.segment_weights, PeriodSchedule):
        schedule(d.segment_weights, "$.demand.segment_weights", len(s.segments))
    else:
        _require(
            set(d.segment_weights) == set(indexes["markets"]),
            "$.demand.segment_weights",
            "requires every market",
        )
        for m, sc in d.segment_weights.items():
            schedule(sc, "$.demand.segment_weights." + m, len(s.segments))
    _require(
        math.isfinite(d.holiday_multiplier) and d.holiday_multiplier > 0,
        "$.demand.holiday_multiplier",
        "must be positive",
    )
    _require(
        0 <= d.arrival_probability_cap <= 1,
        "$.demand.arrival_probability_cap",
        "invalid probability cap",
    )
    for i, seg in enumerate(s.segments):
        _prob(tuple(c.weight for c in seg.classes), f"$.segments[{i}].classes")
        for j, c in enumerate(seg.classes):
            path = f"$.segments[{i}].classes[{j}]"
            _require(
                math.isfinite(c.alpha) and c.alpha > 0,
                path + ".alpha",
                "must be positive",
            )
            _require(
                "*" in c.market_value or set(indexes["markets"]) <= set(c.market_value),
                path + ".market_value",
                "missing market values",
            )
            _require(
                set(c.carrier_pref) <= set(indexes["carriers"]),
                path + ".carrier_pref",
                "unknown carrier",
            )
    n = len(s.exogenous.states)
    _require(
        n > 0 and len({x.name for x in s.exogenous.states}) == n,
        "$.exogenous.states",
        "states must be unique",
    )
    _prob(s.exogenous.initial, "$.exogenous.initial", n)
    matrices = list(s.exogenous.transitions.items())
    if s.exogenous.default_transition is not None:
        matrices.append(("default", s.exogenous.default_transition))
    for t, matrix in matrices:
        _require(len(matrix) == n, "$.exogenous.transitions", "incorrect matrix size")
        for row in matrix:
            _prob(row, "$.exogenous.transitions." + str(t), n)
    exo_names = {x.name for x in s.exogenous.states}
    for i, e in enumerate(s.capacity_process.events):
        _require(
            all(l in indexes["legs"] for l, _ in e.effects),
            f"$.capacity_process.events[{i}].effects",
            "unknown leg",
        )
        if "exo_transition" in e.trigger:
            _require(
                set(e.trigger["exo_transition"]) <= exo_names,
                f"$.capacity_process.events[{i}].trigger",
                "unknown exogenous state",
            )
    _require(
        s.timing.mode in ("simultaneous", "leader_follower"),
        "$.timing.mode",
        "unknown timing mode",
    )
    _require(
        set(s.timing.leaders) <= set(indexes["carriers"]),
        "$.timing.leaders",
        "unknown leader",
    )
    requests = {q.id: q for q in s.quotes.requests}
    _require(
        len(requests) == len(s.quotes.requests),
        "$.quotes.requests",
        "duplicate shopping IDs",
    )
    backgrounds = {b.id: b for b in s.background}
    for i, q in enumerate(s.quotes.requests):
        p = f"$.quotes.requests[{i}]"
        _require(q.observer in indexes["carriers"], p + ".observer", "unknown observer")
        for target in q.target_products:
            _require(
                target in indexes["products"] or target in backgrounds,
                p + ".target_products",
                "unknown target",
            )
            owner = (
                indexes["products"][target].carrier
                if target in indexes["products"]
                else backgrounds[target].carrier
            )
            _require(
                owner != q.observer, p + ".target_products", "cannot shop own products"
            )
            if q.lag_periods == 0:
                _require(
                    s.timing.mode == "leader_follower" and owner in s.timing.leaders,
                    p + ".lag_periods",
                    "current quotes require a leader",
                )
        _require(
            q.lag_periods >= 0 and q.every_k_periods >= 1, p, "invalid quote timing"
        )
        _require(
            q.fare_bin >= 0
            and q.fare_noise_sd >= 0
            and 0 <= q.missing_probability <= 1,
            p,
            "invalid quote noise",
        )
        _require(
            q.availability in ("flag", "seats_capped", "none"),
            p + ".availability",
            "unknown display mode",
        )
    for carrier in s.carriers:
        if any(p.controlled and p.carrier == carrier.code for p in s.products):
            _require(
                sum(p.carrier == carrier.code for p in s.pricing) == 1,
                "$.pricing",
                "one pricing specification per controlled carrier",
            )
            _require(
                sum(p.carrier == carrier.code for p in s.controllers) == 1,
                "$.controllers",
                "one controller per controlled carrier",
            )
    for i, p in enumerate(s.pricing):
        _require(
            p.carrier in indexes["carriers"],
            f"$.pricing[{i}].carrier",
            "unknown carrier",
        )
        _require(
            p.sharing in ("carrier", "market_group", "product"),
            f"$.pricing[{i}].sharing",
            "unknown sharing rule",
        )
        size = len(p.controllable_features)
        blocks = (
            1
            if p.sharing == "carrier"
            else len([j for j in s.products if j.carrier == p.carrier])
            if p.sharing == "product"
            else len({j.market for j in s.products if j.carrier == p.carrier})
        )
        _require(
            len(p.theta_bounds) == size * blocks,
            f"$.pricing[{i}].theta_bounds",
            "incorrect coefficient dimension",
        )
        _require(
            all(
                math.isfinite(a) and math.isfinite(b) and a <= b
                for a, b in p.theta_bounds
            ),
            f"$.pricing[{i}].theta_bounds",
            "invalid bounds",
        )
        for point in p.theta_grid or ():
            _require(
                len(point) == len(p.theta_bounds)
                and all(a <= x <= b for x, (a, b) in zip(point, p.theta_bounds)),
                f"$.pricing[{i}].theta_grid",
                "grid point outside bounds",
            )
    _require(
        len({o.carrier for o in s.objectives}) == len(s.objectives),
        "$.objectives",
        "duplicate objective",
    )
    for i, o in enumerate(s.objectives):
        _require(
            o.carrier in indexes["carriers"]
            and o.profit_scale > 0
            and math.isfinite(o.profit_scale)
            and o.share_weight >= 0,
            f"$.objectives[{i}]",
            "invalid objective",
        )
    for i, a in enumerate(s.information):
        _require(
            all(
                q in requests and requests[q].observer == a.observer for q in a.requests
            ),
            f"$.information[{i}].requests",
            "invalid arm request",
        )
    for pair in s.metadata.get("nested_arms", ()):
        arms = {a.name: a for a in s.information}
        _require(
            all(name in arms for name in pair), "$.metadata.nested_arms", "unknown arm"
        )
        _require(
            set(arms[pair[0]].requests) <= set(arms[pair[1]].requests),
            "$.metadata.nested_arms",
            "arms are not nested",
        )
    for i, b in enumerate(s.background):
        capacity(b.capacity, f"$.background[{i}].capacity")


def validate_exact_compatible(s: MarketSpec) -> None:
    """Reject processes without a supported finite exact representation (Eqs. 1–19)."""
    validate_spec(s)
    _require(
        s.choice.model == "latent_class_mnl",
        "$.choice.model",
        "mixed logit is sampling-only",
    )
    for q in s.quotes.requests:
        _require(
            q.fare_noise_sd == 0 or q.fare_bin > 0,
            "$.quotes.requests",
            "exact noise requires discrete bins",
        )
    allowed = {
        "fixed",
        "schedule",
        "threshold_linear_rule",
        "matching",
        "external",
        "bid_price_dlp",
        "emsr_b",
        "myopic_br",
    }
    for c in s.controllers:
        _require(
            c.controller in allowed,
            "$.controllers",
            "controller has no declared finite exact memory",
        )
        quote_dependent = c.controller in (
            "matching",
            "bid_price_dlp",
            "emsr_b",
            "myopic_br",
        ) or (
            c.controller == "threshold_linear_rule"
            and any(r.get("responses") for r in c.params.get("rules", {}).values())
        )
        if quote_dependent:
            for q in s.quotes.requests:
                if q.observer == c.carrier:
                    _require(
                        q.fare_noise_sd == 0 and q.missing_probability == 0,
                        "$.quotes.requests",
                        "stochastic rival observation/action branches are sampling-only",
                    )
