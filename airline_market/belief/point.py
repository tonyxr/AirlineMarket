"""Certainty-equivalent decisions, retaining a proper underlying filter."""

import numpy as np
from .exact import ExactBelief
from .factorized import inventory_key


class PointBelief:
    """
    Certainty-equivalent planning distribution backed by a proper history filter (Eqs.
    24–25, 30).
    """

    def __init__(self, compiled=None, focal=None, arm=None, *, base=None, mode="map"):
        """
        Bind declared inputs and initialize PointBelief bookkeeping (Eqs. 24–25, 30).
        """
        self.base = base or ExactBelief(compiled, focal, arm)
        self.mode = mode
        self.mass = {}

    def __getattr__(self, name):
        """
        Forward supported interface queries or reject forbidden hidden-state access
        (Eqs. 24–25, 30).
        """
        return getattr(self.base, name)

    def _select(self):
        """
        Choose a feasible MAP or nearest-mean hypothesis for planning only (Eqs. 24–25,
        30).
        """
        states = sorted(self.base.mass)
        if self.mode == "map":
            chosen = max(states, key=lambda h: self.base.mass[h])
        elif self.mode == "mean_round":
            mean = sum(
                (
                    p
                    * np.asarray(
                        inventory_key(h, self.compiled, self.focal), dtype=float
                    )
                    for h, p in self.base.mass.items()
                ),
                np.zeros(len(inventory_key(states[0], self.compiled, self.focal))),
            )
            chosen = min(
                states,
                key=lambda h: float(
                    (
                        (np.asarray(inventory_key(h, self.compiled, self.focal)) - mean)
                        ** 2
                    ).sum()
                ),
            )
        else:
            raise ValueError("point mode must be map or mean_round")
        self.mass = {chosen: 1.0}

    def reset(self, obs):
        """
        Start a new posterior from the first permitted observation (Eqs. 24–25, 30).
        """
        self.base.reset(obs)
        self._select()

    def update(self, before, action, after):
        """
        Condition the maintained posterior on the observed action/outcome sequence (Eqs.
        24–25, 30).
        """
        self.base.update(before, action, after)
        self._select()

    def sample(self, n, rng):
        """
        Draw model hypotheses from the maintained posterior using the supplied RNG (Eqs.
        24–25, 30).
        """
        return [next(iter(self.mass))] * n

    def key(self):
        """
        Return the deterministic memoization key for the represented state (Eqs. 24–25,
        30).
        """
        return tuple(self.mass.items())

    def entropy(self):
        """
        Return Shannon entropy of the represented hidden-state distribution (Eqs. 24–25,
        30).
        """
        return 0.0

    def leg_marginals(self):
        """
        Return each rival leg's probability vector over residual seats (Eqs. 24–25, 30).
        """
        return ExactBelief.leg_marginals(self)

    def count_marginals(self):
        """
        Return each rival product's posterior active-booking count distribution (Eqs.
        24–25, 30).
        """
        return ExactBelief.count_marginals(self)

    def event_probability(self, event):
        """
        Integrate a declared joint event over the maintained posterior (Eqs. 24–25, 30).
        """
        return ExactBelief.event_probability(self, event)

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 24–25, 30).
        """
        return self.base.snapshot(), dict(self.mass)

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 24–25, 30)."""
        self.base.restore(s[0])
        self.mass = dict(s[1])
