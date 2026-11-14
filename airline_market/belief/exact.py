"""Finite Bayesian filtering through the shared exact kernel (Eqs. 24–25)."""

from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import math
import numpy as np
from ..core.errors import ZeroLikelihoodError
from ..sim.kernel import ExactKernel
from .base import HiddenState


def ledger_key(obs):
    """
    Extract the observable active booking ledger for likelihood matching (Eqs. 24–25,
    30).
    """
    return tuple((b.product, b.fare, b.period) for b in obs.own_bookings)


class ExactBelief:
    """
    Finite Bayesian filter using only own outcomes and permitted shopping signals (Eqs.
    24–25, 30).
    """

    def __init__(self, compiled, focal: str, arm: str, kernel=None):
        """
        Bind declared inputs and initialize ExactBelief bookkeeping (Eqs. 24–25, 30).
        """
        self.compiled = compiled
        self.focal = compiled.carrier_index[focal]
        self.arm = next(
            a
            for a in compiled.spec.information
            if a.name == arm and a.observer == focal
        )
        from ..sim.fast_kernel import make_exact_kernel

        self.kernel = kernel or make_exact_kernel(compiled, focal, self.arm)
        self.mass = {}
        self.history = []
        self.last_observation = None
        self.track_background = any(
            target in {b.id for b in compiled.spec.background}
            for request in compiled.spec.quotes.requests
            if request.id in self.arm.requests or request.observer != focal
            for target in request.target_products
        )

    def _hidden(self, ks):
        """
        Separate hidden hypotheses from the known focal-carrier state (Eqs. 24–25, 30).
        """
        if not self.track_background:
            ks = replace(ks, background=tuple(0 for _ in ks.background))
        rivals = tuple(x for i, x in enumerate(ks.carriers) if i != self.focal)
        latent = None if self.compiled.spec.exogenous.states[ks.exo].public else ks.exo
        return HiddenState(rivals, ks.background, latent, ks.history, ks)

    def _normalize(self, weights):
        """
        Normalize finite positive mass; reject an impossible observation (Eqs. 24–25,
        30).
        """
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ZeroLikelihoodError(
                "observation has zero likelihood under the declared model"
            )
        self.mass = {h: p / total for h, p in weights.items() if p > 0}

    def _privileged_match(self, obs, ks):
        """
        Condition oracle data on current capacities, counts and controller memory (Eqs.
        24–25, 30).
        """
        truth = obs.privileged
        if truth is None:
            return True
        # Stored hypotheses precede leader posting. Privileged observations
        # include the deterministic memory update made by that posting.
        predicted = (
            self.kernel.observation(ks, self.arm).privileged
            if hasattr(self.kernel, "observation")
            else self.kernel.engine(ks, self.focal)
            .observe(self.focal, self.arm)
            .privileged
        )
        for i, state in enumerate(ks.carriers):
            if state.capacity != truth.capacities[i]:
                return False
            counts = tuple(
                sum(b.product == j for b in truth.bookings[i])
                for j in self.compiled.products_of[i]
            )
            if counts != state.counts:
                return False
            if truth.controller_memory[i] != predicted.controller_memory[i]:
                return False
        return truth.exo == ks.exo

    def reset(self, first_obs):
        """
        Start a new posterior from the first permitted observation (Eqs. 24–25, 30).
        """
        weights = defaultdict(float)
        for p, ks in self.kernel.initial_distribution(self.focal):
            if ks.carriers[self.focal].capacity != first_obs.own_capacity:
                continue
            if not self._privileged_match(first_obs, ks):
                continue
            for po, obs in self.kernel.observation_distribution(
                ks, self.focal, self.arm, first_obs.quotes
            ):
                if obs.public_exo != first_obs.public_exo:
                    continue
                h = self._hidden(ks)
                weights[h] += p * po
        self._normalize(weights)
        self.last_observation = first_obs
        self.history = []

    def conditional_successors(self, h, before, action, after):
        """
        Enumerate next hidden states consistent with the observed own outcome (Eqs.
        24–25, 30).
        """
        result = defaultdict(float)
        for q, ks, signal in self.kernel.transition(
            h.kernel,
            self.focal,
            action,
            arm=self.arm,
            observation=before,
            observed_quotes=after.quotes,
        ):
            if (
                signal.purchases != after.own_purchases
                or signal.own_capacity != after.own_capacity
            ):
                continue
            if signal.public_exo != tuple(sorted(after.public_exo.items())):
                continue
            if abs(signal.cash_delta - (after.own_cash - before.own_cash)) > 1e-9:
                continue
            if ledger_key(signal.observation) != ledger_key(after):
                continue
            if not self._privileged_match(after, ks):
                continue
            result[self._hidden(ks)] += q
        return dict(result)

    def update(self, before, action, after):
        """
        Condition the maintained posterior on the observed action/outcome sequence (Eqs.
        24–25, 30).
        """
        weights = defaultdict(float)
        for h, p in self.mass.items():
            for successor, q in self.conditional_successors(
                h, before, action, after
            ).items():
                weights[successor] += p * q
        self._normalize(weights)
        self.last_observation = after
        self.history.append((before, action, after))

    def sample(self, n: int, rng):
        """
        Draw model hypotheses from the maintained posterior using the supplied RNG (Eqs.
        24–25, 30).
        """
        states, weights = zip(*sorted(self.mass.items()))
        cdf = np.cumsum(weights)
        return [
            states[min(int(np.searchsorted(cdf, u, side="right")), len(states) - 1)]
            for u in rng.random(n)
        ]

    def leg_marginals(self):
        """
        Return each rival leg's probability vector over residual seats (Eqs. 24–25, 30).
        """
        c = self.compiled
        result = {}
        rival_indexes = [i for i in range(len(c.spec.carriers)) if i != self.focal]
        for r, i in enumerate(rival_indexes):
            for l in range(len(c.legs_of[i])):
                values = {}
                for h, p in self.mass.items():
                    state = h.rivals[r]
                    x = state.capacity[l] - sum(
                        int(c.A[i][l, j]) * n for j, n in enumerate(state.counts)
                    )
                    values[x] = values.get(x, 0.0) + p
                arr = np.zeros(max(values, default=0) + 1)
                for x, p in values.items():
                    arr[x] = p
                result[(i, l)] = arr
        return result

    def count_marginals(self):
        """
        Return each rival product's posterior active-booking count distribution (Eqs.
        24–25, 30).
        """
        c = self.compiled
        result = {}
        for r, i in enumerate(
            k for k in range(len(c.spec.carriers)) if k != self.focal
        ):
            for j in range(len(c.products_of[i])):
                maxn = max((h.rivals[r].counts[j] for h in self.mass), default=0)
                arr = np.zeros(maxn + 1)
                for h, p in self.mass.items():
                    arr[h.rivals[r].counts[j]] += p
                result[(i, j)] = arr
        return result

    def event_probability(self, event):
        """
        Integrate a declared joint event over the maintained posterior (Eqs. 24–25, 30).
        """
        return sum(
            p * event.evaluate(h, self.compiled, self.focal)
            for h, p in self.mass.items()
        )

    def entropy(self):
        """
        Return Shannon entropy of the represented hidden-state distribution (Eqs. 24–25,
        30).
        """
        return -sum(p * math.log(p) for p in self.mass.values())

    def key(self):
        """
        Return the deterministic memoization key for the represented state (Eqs. 24–25,
        30).
        """
        return tuple(sorted(self.mass.items()))

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 24–25, 30).
        """
        return (dict(self.mass), list(self.history), self.last_observation)

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 24–25, 30)."""
        self.mass, self.history, self.last_observation = dict(s[0]), list(s[1]), s[2]
