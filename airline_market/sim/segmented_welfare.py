"""A.10/B0/V9: conditional logit surplus and read-only ordinary-request strata.

Analyst only. Re-entry attempts are not additional independent welfare people.
Whole-passenger disruption welfare, denied demand and first-best allocation are
unidentified by the ordinary request log and remain explicitly unmeasured.
"""

import math
from collections import defaultdict
from itertools import product

from airline_market.sim.accounting import carrier_metrics


def conditional_logit_surplus(utilities, alpha, *, outside_index=-1):
    """Expected-max gain relative to outside, conditional on one taste/choice set.

    For iid unit-scale Gumbels, Euler constants cancel: (logsumexp(u)-u_out)/alpha.
    Unavailable alternatives are -infinity. No tastes or physical cohorts sampled.
    """
    values = tuple(float(x) for x in utilities)
    alpha = float(alpha)
    if not values or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("nonempty utilities and finite positive alpha required")
    if type(outside_index) is not int or not -len(values) <= outside_index < len(
        values
    ):
        raise ValueError("explicit valid outside index required")
    if any(math.isnan(x) or x == math.inf for x in values):
        raise ValueError("finite or unavailable (-infinity) utilities required")
    outside = values[outside_index]
    if not math.isfinite(outside):
        raise ValueError("finite outside utility required")
    centered = tuple(x - outside for x in values)
    peak = max(centered)
    return (peak + math.log(math.fsum(math.exp(x - peak) for x in centered))) / alpha


def conditional_mixture_surplus(rows):
    """Exact weighted conditional-class surplus; integrate each class's own alpha.

    Rows are (weight, utilities, alpha). These are supplied conditional finite
    classes, not an integration rule or fallback for continuous mixed logit.
    """
    rows = tuple(rows)
    if not rows:
        raise ValueError("explicit conditional class rows required")
    weights = tuple(float(row[0]) for row in rows)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(
        math.fsum(weights), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("nonnegative finite weights must sum to one; no normalization")
    return math.fsum(
        w * conditional_logit_surplus(u, a) for w, (_, u, a) in zip(weights, rows)
    )


def ordinary_welfare_summary(engine, *, market_groups, early_stop):
    """Read existing native request logs; never call actors, RNG or simulation.

    market_groups explicitly labels request O&Ds as local/connecting/mixed. Missing
    labels stay unclassified; groups do not depend on which ticket was purchased.
    early_stop is an exclusive native-period boundary, with no day/hour inference.
    Empty strata mean zero *logged* requests in the observed native prefix only.
    """
    c, world = engine.compiled, engine.world
    if world is None:
        raise ValueError("native world must be initialized")
    T = c.spec.horizon.n_periods
    if type(early_stop) is not int or not 0 <= early_stop <= T:
        raise ValueError("explicit native early/late boundary required")
    labels = dict(market_groups)
    ids = {m.id for m in c.spec.markets}
    allowed = ("local", "connecting", "mixed", "unclassified")
    if set(labels) - ids or any(v not in allowed for v in labels.values()):
        raise ValueError(
            "known O&D IDs and declared local/connecting/mixed/unclassified labels required"
        )
    segments = tuple(s.id for s in c.spec.segments)
    buckets = defaultdict(list)
    logs = tuple(world.request_log)
    for row in logs:
        t, m, s = row["period"], row["market"], row["segment"]
        if any(type(x) is not int for x in (t, m, s)) or not (
            0 <= t < world.t and 0 <= m < len(c.spec.markets) and 0 <= s < len(segments)
        ):
            raise ValueError("request coordinates outside completed native prefix")
        if not all(
            math.isfinite(float(row[k]))
            for k in ("consumer_surplus", "realized_surplus")
        ):
            raise ValueError(
                "finite logged surplus required; missing values not zero-filled"
            )
        if type(row["unavailable"]) is not bool:
            raise ValueError("explicit native availability indicator required")
        key = (
            segments[s],
            labels.get(c.spec.markets[m].id, "unclassified"),
            "early" if t < early_stop else "late",
        )
        buckets[key].append(row)

    def total(rows):
        return {
            "logged_requests": len(rows),
            "outside_choices": sum(r["chosen"] is None for r in rows),
            "outside_with_unavailable_option": sum(
                r["chosen"] is None and r["unavailable"] for r in rows
            ),
            "conditional_consumer_surplus": math.fsum(
                float(r["consumer_surplus"]) for r in rows
            ),
            "realized_surplus": math.fsum(float(r["realized_surplus"]) for r in rows),
        }

    joint = [
        {
            "segment": s,
            "request_market_group": g,
            "booking_period_group": b,
            **total(buckets[(s, g, b)]),
        }
        for s, g, b in product(segments, allowed, ("early", "late"))
    ]
    profits = {
        carrier.code: carrier_metrics(world, c, i)["profit"]
        for i, carrier in enumerate(c.spec.carriers)
    }
    ordinary = total(logs)
    if hasattr(engine, "_passengers"):
        stranded = len(engine._passengers)
    else:
        stranded = len(getattr(engine, "stranded", ()))
    return {
        "plan_id": "A.10/B0/V9",
        "scope": "Analyst conditional ordinary-request choice-set welfare on completed native prefix; no re-entry search CS added",
        "observed_native_periods": world.t,
        "horizon_periods": T,
        "early_stop_exclusive": early_stop,
        "labels_complete": set(labels) == ids
        and all(v != "unclassified" for v in labels.values()),
        "ordinary": ordinary,
        "joint_strata": joint,
        "by_segment": [
            {"segment": s, **total([r for r in logs if segments[r["segment"]] == s])}
            for s in segments
        ],
        "by_market": [
            {
                "market": m.id,
                "request_market_group": labels.get(m.id, "unclassified"),
                **total([r for r in logs if r["market"] == i]),
            }
            for i, m in enumerate(c.spec.markets)
        ],
        "carrier_profits": profits,
        "ordinary_CS_plus_lambda_profit_index": [
            {
                "lambda": w,
                "value": ordinary["conditional_consumer_surplus"]
                + w * math.fsum(profits.values()),
            }
            for w in (0.5, 1.0)
        ],
        "stranded_lineages_observed": stranded,
        "stranded_lifecycle_consumer_surplus": None,
        "total_lifecycle_surplus": None,
        "denied_requests": None,
        "misallocation": None,
        "coordination_index": None,
        "primary_B4_welfare_adopted": False,
        "qualification": "Outside with an unavailable option is a native diagnostic, not identified denied demand. Original ordinary rows remain ordinary even if that traveller is later stranded. Re-entry histories are not summed as extra welfare passengers; no waiting/departure utility inferred.",
    }
