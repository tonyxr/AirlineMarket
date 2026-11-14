"""Cohort contribution, journey share, and choice-set welfare (Eqs. 20–22)."""

from collections import defaultdict
import numpy as np
from .ledger import inventory


def carrier_metrics(world, compiled, i: int):
    """
    Compute one carrier's profit, competitive journey share and capacity use (Eqs.
    3–16).
    """
    s = compiled.spec
    state = world.carriers[i]
    code = s.carriers[i].code
    objective = next(o for o in s.objectives if o.carrier == code)
    markets = set(objective.share_markets or [m.id for m in s.markets if m.in_segment])
    journeys = [
        sum(s.products[b.product].market in markets for b in c.bookings)
        for c in world.carriers
    ]
    service = sum(float(compiled.product_cost[b.product]) for b in state.bookings)
    profit = state.cash - service
    share = journeys[i] / sum(journeys) if sum(journeys) else 0.0
    initial = [world.initial_capacity[l] for l in compiled.legs_of[i]]
    inv = inventory(state, compiled, i)
    loads = [
        (int(k) - int(x)) / max(1, initial[l])
        for l, (k, x) in enumerate(zip(state.capacity, inv))
    ]
    return {
        "revenue": state.revenue,
        "refunds": state.refunds,
        "penalties": state.penalties,
        "canceled": state.canceled,
        "cancellations": float(state.canceled),
        "service_cost": service,
        "profit": profit,
        "fixed_cost": 0.0,
        "journeys": float(journeys[i]),
        "share": share,
        "objective": profit / objective.profit_scale + objective.share_weight * share,
        "load_factor": loads,
        "sellout_period": [world.sellout_period.get(l) for l in compiled.legs_of[i]],
        "inventory": list(map(int, inv)),
    }


def metrics(world, compiled):
    """Return analyst-only cohort accounting and welfare summaries (Eqs. 3–16)."""
    carriers = {
        c.code: carrier_metrics(world, compiled, i)
        for i, c in enumerate(compiled.spec.carriers)
    }
    focal = compiled.spec.metadata.get("focal", compiled.spec.carriers[0].code)
    own = carriers[focal]
    other = [c.code for c in compiled.spec.carriers if c.code != focal]
    logs = world.request_log
    outside = sum(row["chosen"] is None for row in logs)
    markets = {}
    for m, market in enumerate(compiled.spec.markets):
        rows = [r for r in logs if r["market"] == m]
        by_carrier = {}
        for i, carrier in enumerate(compiled.spec.carriers):
            bookings = [
                b
                for b in world.carriers[i].bookings
                if compiled.product_market[b.product] == m
            ]
            fares = [b.fare for b in bookings]
            by_carrier[carrier.code] = {
                "journeys": len(bookings),
                "median_fare": float(np.median(fares)) if fares else None,
                "fare_dispersion": float(
                    (np.quantile(fares, 0.75) - np.quantile(fares, 0.25))
                    / max(1e-9, np.median(fares))
                )
                if fares
                else None,
                "nonstop_journeys": sum(
                    len(compiled.spec.products[b.product].legs) == 1 for b in bookings
                ),
            }
        markets[market.id] = {
            "requests": len(rows),
            "outside": sum(r["chosen"] is None for r in rows),
            "consumer_surplus": sum(r["consumer_surplus"] for r in rows),
            "carriers": by_carrier,
        }
    result = {
        **{
            key: own[key]
            for key in (
                "profit",
                "share",
                "objective",
                "journeys",
                "refunds",
                "cancellations",
            )
        },
        "rival_profit": sum(carriers[c]["profit"] for c in other),
        "rival_segment_journeys": sum(carriers[c]["journeys"] for c in other),
        "consumer_surplus": sum(r["consumer_surplus"] for r in logs),
        "realized_consumer_surplus": sum(r["realized_surplus"] for r in logs),
        "spill": sum(r["chosen"] is None and r["unavailable"] for r in logs),
        "no_purchase_rate": outside / len(logs) if logs else 0.0,
        "requests": len(logs),
        "carriers": carriers,
        "markets": markets,
        "background": {
            b.id: {"stock": x.stock, "revenue": x.revenue}
            for b, x in zip(compiled.spec.background, world.background)
        },
    }
    return result
