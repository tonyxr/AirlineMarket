"""Network deterministic-LP bid-price baseline, labeled a distinct price class."""

import numpy as np
from scipy.optimize import linprog
from ..core.types import PriceAction
from ..core.money import round_fare, clip_fare
from .base import StatelessController
from .forecast import remaining_demand


def solve_dlp(A, capacity, revenues, demand):
    """
    Solve the network capacity LP and return rounded leg opportunity costs (Eqs. 18–19).
    """
    result = linprog(
        -np.asarray(revenues, dtype=float),
        A_ub=np.asarray(A, dtype=float),
        b_ub=np.asarray(capacity, dtype=float),
        bounds=[(0, float(x)) for x in demand],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"network DLP failed: {result.message}")
    return result.x, -result.ineqlin.marginals, float(-result.fun)


class DLPBidPriceController(StatelessController):
    """
    Observable network revenue-management baseline using capacity duals (Eqs. 18–19).
    """

    policy_class = "price"
    exact_compatible = True

    def __init__(self, reopt_every: int = 1, markup=0.0):
        """
        Bind declared inputs and initialize DLPBidPriceController bookkeeping (Eqs.
        18–19).
        """
        self.reopt_every = reopt_every
        self.markup = markup
        self.cached = None
        self.last = -1
        self.duals = ()

    def reset(self, ctx, rng):
        """Start a new cohort and clear per-episode memory (Eqs. 18–19)."""
        super().reset(ctx, rng)
        self.cached = None
        self.last = -1
        self.duals = ()

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        c = self.ctx.compiled
        i = self.carrier
        if self.cached is None or obs.t - self.last >= self.reopt_every:
            refs = np.array(
                [c.spec.products[j].reference_fare for j in c.products_of[i]]
            )
            demand = remaining_demand(c, i, obs)
            _, duals, _ = solve_dlp(
                c.A[i],
                obs.own_inventory,
                refs - c.product_cost[list(c.products_of[i])],
                demand,
            )
            self.duals = tuple(round_fare(x) for x in duals)
            prices = {}
            for local, j in enumerate(c.products_of[i]):
                p = c.spec.products[j]
                mark = (
                    self.markup.get(p.id, 0.0)
                    if hasattr(self.markup, "get")
                    else self.markup
                )
                fare = max(p.reference_fare, float(c.A[i][:, local] @ duals) + mark)
                prices[j] = round_fare(clip_fare(fare, p.fare_floor, p.fare_cap))
            self.cached = prices
            self.last = obs.t
        return PriceAction(self.cached)

    def memory(self):
        """
        Return the controller's hashable finite memory, without world references (Eqs.
        18–19).
        """
        return (self.last, tuple(sorted((self.cached or {}).items())), self.duals)

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 18–19).
        """
        return self.memory()

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 18–19)."""
        self.last, values, self.duals = s if s else (-1, (), ())
        self.cached = dict(values) if values else None
