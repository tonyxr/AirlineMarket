"""EMSR-b normal-demand leg baseline with prorated connecting fare ladders."""

import math
import numpy as np
from scipy.special import ndtri
from ..core.types import PriceAction
from ..core.money import round_fare, clip_fare
from .base import StatelessController
from .forecast import remaining_demand


def emsr_b_protection(fares, means, stds, capacity=math.inf):
    """
    Compute the EMSR-b protection level for aggregated higher-fare demand (Eqs. 18–19).
    """
    order = np.argsort(-np.asarray(fares))
    f = np.asarray(fares)[order]
    mu = np.asarray(means)[order]
    sd = np.asarray(stds)[order]
    protections = [0.0]
    for k in range(1, len(f)):
        total = float(mu[:k].sum())
        average = float((f[:k] * mu[:k]).sum() / total) if total else float(f[0])
        ratio = min(1.0, max(0.0, 1 - f[k] / average))
        y = total + float(np.sqrt((sd[:k] ** 2).sum())) * float(
            ndtri(np.clip(ratio, 1e-12, 1 - 1e-12))
        )
        protections.append(min(capacity, max(0.0, y)))
    return tuple(protections)


class EMSRbController(StatelessController):
    """Leg-based fare-ladder protection baseline (Eqs. 18–19)."""

    policy_class = "fare_ladder"
    exact_compatible = True

    def __init__(self, ladders=None, forecast=None):
        """
        Bind declared inputs and initialize EMSRbController bookkeeping (Eqs. 18–19).
        """
        self.ladders = ladders or {}
        self.forecast = forecast or {}

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        c = self.ctx.compiled
        i = self.carrier
        expected = remaining_demand(c, i, obs)
        prices = {}
        ladders = {}
        pools = {l: [] for l in c.legs_of[i]}
        for local, j in enumerate(c.products_of[i]):
            p = c.spec.products[j]
            ladder = tuple(
                sorted(
                    self.ladders.get(
                        p.id,
                        (
                            p.reference_fare * 1.25,
                            p.reference_fare,
                            p.reference_fare * 0.75,
                        ),
                    ),
                    reverse=True,
                )
            )
            ladders[j] = ladder
            mean = self.forecast.get(p.id, {}).get(
                "means", [expected[local] / len(ladder)] * len(ladder)
            )
            std = self.forecast.get(p.id, {}).get(
                "stds", [math.sqrt(max(x, 0.0)) for x in mean]
            )
            distances = [c.spec.legs[c.leg_index[l]].distance_mi or 1.0 for l in p.legs]
            distance = sum(distances)
            for leg, d in zip(p.legs, distances):
                pools[c.leg_index[leg]].extend(
                    (fare * d / distance, mu, sd)
                    for fare, mu, sd in zip(ladder, mean, std)
                )
        for j, ladder in ladders.items():
            p = c.spec.products[j]
            distances = [c.spec.legs[c.leg_index[l]].distance_mi or 1.0 for l in p.legs]
            distance = sum(distances)
            open_class = len(ladder) - 1
            for leg, d in zip(p.legs, distances):
                glob = c.leg_index[leg]
                loc = c.legs_of[i].index(glob)
                protection = []
                for fare in ladder:
                    prorated = fare * d / distance
                    higher = [row for row in pools[glob] if row[0] > prorated]
                    if not higher:
                        protection.append(0.0)
                        continue
                    total = sum(row[1] for row in higher)
                    average = (
                        sum(row[0] * row[1] for row in higher) / total
                        if total
                        else prorated
                    )
                    sd = math.sqrt(sum(row[2] ** 2 for row in higher))
                    q = np.clip(1 - prorated / max(average, 1e-12), 1e-12, 1 - 1e-12)
                    protection.append(
                        min(
                            obs.own_capacity[loc],
                            max(0.0, total + sd * float(ndtri(q))),
                        )
                    )
                eligible = [
                    k for k, y in enumerate(protection) if obs.own_inventory[loc] > y
                ]
                open_class = min(open_class, max(eligible) if eligible else 0)
            prices[j] = round_fare(
                clip_fare(ladder[open_class], p.fare_floor, p.fare_cap)
            )
        return PriceAction(prices)
