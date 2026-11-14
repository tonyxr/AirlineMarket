"""Feasible product-of-inventory-marginals compression (Eq. 30)."""

from collections import defaultdict
from dataclasses import replace
from itertools import product
import math
import numpy as np
from ..core.errors import KernelLimitError
from .exact import ExactBelief


def inventory_key(hidden, compiled, focal):
    """
    Project a hidden hypothesis onto the vector of rival residual leg seats (Eqs. 24–25,
    30).
    """
    values = []
    for state, i in zip(
        hidden.rivals, [i for i in range(len(compiled.spec.carriers)) if i != focal]
    ):
        values.extend(
            state.capacity[l]
            - sum(int(compiled.A[i][l, j]) * n for j, n in enumerate(state.counts))
            for l in range(len(state.capacity))
        )
    return tuple(values)


def _reconstruct_template(belief, target, template):
    """
    Enumerate feasible count ledgers for stock targets and a nuisance template (Eqs.
    24–25, 30).
    """
    c = belief.compiled
    ks = template.kernel
    options = []
    offset = 0
    for r, i in enumerate(k for k in range(len(c.spec.carriers)) if k != belief.focal):
        source = ks.carriers[i]
        nlegs = len(source.capacity)
        stock = target[offset : offset + nlegs]
        offset += nlegs
        support = []
        for l in c.legs_of[i]:
            leg = c.spec.legs[l]
            values = set(leg.capacity.support)
            for prior in c.spec.capacity_priors:
                if leg.id in prior.legs:
                    col = prior.legs.index(leg.id)
                    values = {v[col] for v in prior.support}
            support.append(sorted(values))
        candidates = []
        for capacity in product(*support):
            if any(x > k for x, k in zip(stock, capacity)):
                continue
            limits = [
                min([capacity[l] for l in range(nlegs) if c.A[i][l, j]])
                for j in range(len(source.counts))
            ]
            own_count = sum(ks.carriers[belief.focal].counts)
            count_budget = (
                max(0, ks.t - own_count)
                if c.spec.demand.mode == "single_request_bernoulli"
                else sum(capacity)
            )
            for counts in product(*(range(min(k, count_budget) + 1) for k in limits)):
                if sum(counts) > count_budget:
                    continue
                if tuple(np.asarray(capacity) - c.A[i] @ np.asarray(counts)) != stock:
                    continue
                order = tuple(j for j, n in enumerate(counts) for _ in range(n))
                candidate = replace(
                    source,
                    capacity=tuple(capacity),
                    counts=tuple(counts),
                    fares=None,
                    cash=sum(
                        c.spec.products[j].reference_fare * n
                        for j, n in zip(c.products_of[i], counts)
                    ),
                    booking_order=order,
                    periods=tuple((-1,) * n for n in counts),
                )
                candidates.append(candidate)
        options.append((i, candidates))
    result = []
    for combination in product(*(x[1] for x in options)):
        if (
            c.spec.demand.mode == "single_request_bernoulli"
            and sum(sum(state.counts) for state in combination)
            + sum(ks.carriers[belief.focal].counts)
            > ks.t
        ):
            continue
        carriers = list(ks.carriers)
        for (i, _), state in zip(options, combination):
            carriers[i] = state
        result.append(belief._hidden(replace(ks, carriers=tuple(carriers))))
        if len(result) > belief.kernel.max_outcomes:
            raise KernelLimitError(
                "feasible factorization support exceeds max_outcomes"
            )
    return result


def _reconstruct(belief, target):
    """
    Combine feasible reconstructions across represented nuisance states (Eqs. 24–25,
    30).
    """
    # Retain every represented nuisance-state configuration. Choosing one
    # arbitrary latent exogenous state here would manufacture information.
    from ..core.hashing import canonical_json

    templates = {}
    for h in belief.mass:
        ks = h.kernel
        nuisance = (
            ks.exo,
            ks.background,
            ks.history,
            ks.events_fired,
            ks.phase,
            ks.posted_leaders,
            tuple(
                (r.previous_prices, r.controller_memory, r.theta, r.posted_prices)
                for r in h.rivals
            ),
        )
        templates.setdefault(canonical_json(nuisance), h)
    states = {}
    for key in sorted(templates):
        for h in _reconstruct_template(belief, target, templates[key]):
            states[h] = None
        if len(states) > belief.kernel.max_outcomes:
            raise KernelLimitError("factorized nuisance support exceeds max_outcomes")
    return list(states)


def compressed_mass(belief):
    """
    Form product leg marginals and record physically infeasible dropped mass (Eqs.
    24–25, 30).
    """
    grouped = defaultdict(dict)
    marginals = []
    for h, p in belief.mass.items():
        grouped[inventory_key(h, belief.compiled, belief.focal)][h] = p
    if not grouped:
        return {}, 0.0
    for col in range(len(next(iter(grouped)))):
        values = defaultdict(float)
        for x, states in grouped.items():
            values[x[col]] += sum(states.values())
        marginals.append(dict(values))
    weights = defaultdict(float)
    truncated = 0.0
    for x in product(*(m.keys() for m in marginals)):
        probability = math.prod(m[value] for m, value in zip(marginals, x))
        if x in grouped:
            total = sum(grouped[x].values())
            for h, p in grouped[x].items():
                weights[h] += probability * p / total
        else:
            states = _reconstruct(belief, x)
            if not states:
                truncated += probability
                continue
            for h in states:
                weights[h] += probability / len(states)
    return dict(weights), truncated


class FactorizedBelief(ExactBelief):
    """Feasible product-of-leg-marginals assumed-density filter (Eqs. 24–25, 30)."""

    def project(self):
        """
        Project onto feasible inventory marginals and measure any truncation distortion
        (Eqs. 24–25, 30).
        """
        original = self.leg_marginals()
        weights, self.truncated_mass = compressed_mass(self)
        self._normalize(weights)
        updated = self.leg_marginals()
        self.max_marginal_deviation = 0.0
        for leg, p in original.items():
            q = updated[leg]
            size = max(len(p), len(q))
            self.max_marginal_deviation = max(
                self.max_marginal_deviation,
                float(
                    np.max(
                        abs(
                            np.pad(p, (0, size - len(p)))
                            - np.pad(q, (0, size - len(q)))
                        )
                    )
                ),
            )

    def reset(self, obs):
        """
        Start a new posterior from the first permitted observation (Eqs. 24–25, 30).
        """
        super().reset(obs)
        self.project()

    def update(self, before, action, after):
        """
        Condition the maintained posterior on the observed action/outcome sequence (Eqs.
        24–25, 30).
        """
        super().update(before, action, after)
        self.project()


def factorize(belief):
    """
    Compress a belief while preserving conditional nuisance information where defined
    (Eqs. 24–25, 30).
    """
    result = FactorizedBelief(
        belief.compiled,
        belief.compiled.spec.carriers[belief.focal].code,
        belief.arm.name,
        belief.kernel,
    )
    result.mass = dict(belief.mass)
    result.last_observation = belief.last_observation
    result.history = list(belief.history)
    result.project()
    return result
