"""Ledger-consistent SMC with exact conditional one-period proposals."""

from collections import defaultdict
from dataclasses import replace
import math
import numpy as np
from ..core.rng import StreamRegistry
from ..core.errors import ZeroLikelihoodError, KernelLimitError
from ..sim.engine import MarketEngine
from ..sim.quotes import posted_quotes
from ..sim.exogenous import public_signal, categorical
from .exact import ExactBelief


class ParticleBelief(ExactBelief):
    """
    Weighted ledger-consistent SMC with conditional proposals and ESS diagnostics (Eqs.
    24–25, 30).
    """

    def __init__(
        self,
        compiled,
        focal,
        arm,
        *,
        n_particles=2000,
        seed=0,
        ess_fraction=0.5,
        strict=True,
        rejuvenation_steps=0,
        kernel=None,
        inner_draws=128,
        conditional_sampling="systematic",
    ):
        """
        Bind declared inputs and initialize ParticleBelief bookkeeping (Eqs. 24–25, 30).
        """
        from ..sim.kernel import ExactKernel
        from ..spec.validate import validate_exact_compatible

        try:
            validate_exact_compatible(compiled.spec)
            self.exact_proposals = True
        except ValueError:
            self.exact_proposals = False
        super().__init__(
            compiled, focal, arm, kernel or ExactKernel(compiled, validate=False)
        )
        if n_particles < 1:
            raise ValueError("n_particles must be positive")
        if not strict:
            raise ValueError(
                "prior-rejection recovery is not enabled; use strict filtering"
            )
        self.n_particles = n_particles
        self.inner_draws = inner_draws
        self.rng = StreamRegistry(seed, "plan/belief").generator()
        if conditional_sampling not in ("systematic", "iid"):
            raise ValueError("conditional_sampling must be systematic or iid")
        self.conditional_sampling = conditional_sampling
        self.ess_fraction = ess_fraction
        self.strict = strict
        self.rejuvenation_steps = rejuvenation_steps
        self.particles = []
        self.weights = np.array([])
        self.diagnostics = []
        self.reference = None

    def conditional_successors(self, h, before, action, after):
        """
        Enumerate next hidden states consistent with the observed own outcome (Eqs.
        24–25, 30).
        """
        if self.exact_proposals:
            try:
                return super().conditional_successors(h, before, action, after)
            except KernelLimitError:
                pass
        # Conditional simulation estimates the joint focal-signal/quote mass.
        # Only fresh planning draws are used; continuous quote likelihoods are
        # integrated analytically rather than requiring equal noisy fares.
        from ..sim.tape import ExogenousTape
        from ..core.types import CoefficientAction, PriceAction
        from .exact import ledger_key

        result = defaultdict(float)
        for _ in range(self.inner_draws):
            e = self.kernel.engine(h.kernel, self.focal)
            e.tape = ExogenousTape.generate(
                self.compiled, int(self.rng.integers(2**63))
            )
            effective = action
            if isinstance(action, CoefficientAction):
                theta = e.world.carriers[self.focal].theta
                update = next(
                    p.update_every
                    for p in self.compiled.spec.pricing
                    if p.carrier == self.compiled.spec.carriers[self.focal].code
                )
                theta = (
                    theta if theta is not None and before.t % update else action.theta
                )
                e.world.carriers[self.focal].theta = theta
                prices = e.rules[self.focal].prices(theta, before)
                effective = PriceAction(
                    dict(zip(self.compiled.products_of[self.focal], prices))
                )
            cash = e.world.carriers[self.focal].cash
            step = e.step({self.focal: effective})
            ks = self.kernel.encode(e.world, self.focal)
            obs = e.observe(self.focal, self.arm)
            if (
                obs.own_purchases != after.own_purchases
                or obs.own_capacity != after.own_capacity
                or obs.public_exo != after.public_exo
            ):
                continue
            if abs(
                obs.own_cash - cash - (after.own_cash - before.own_cash)
            ) > 1e-9 or ledger_key(obs) != ledger_key(after):
                continue
            if not self._privileged_match(after, ks):
                continue
            likelihood = sum(
                p
                for p, _ in self.kernel.observation_distribution(
                    ks, self.focal, self.arm, after.quotes
                )
            )
            result[self._hidden(ks)] += likelihood / self.inner_draws
        self._used_simulated = True
        return dict(result)

    def _aggregate(self):
        """
        Combine duplicate weighted particles into an inspectable posterior mass map
        (Eqs. 24–25, 30).
        """
        mass = defaultdict(float)
        for h, p in zip(self.particles, self.weights):
            if p:
                mass[h] += float(p)
        self._normalize(mass)

    def _resample(self):
        """
        Systematically resample particles when their effective sample size is low (Eqs.
        24–25, 30).
        """
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        indexes = np.minimum(
            np.searchsorted(np.cumsum(self.weights), positions, side="right"),
            self.n_particles - 1,
        )
        self.particles = [self.particles[int(k)] for k in indexes]
        self.weights = np.full(self.n_particles, 1 / self.n_particles)

    def reset(self, first_obs):
        """
        Start a new posterior from the first permitted observation (Eqs. 24–25, 30).
        """
        c = self.compiled
        s = c.spec
        N = self.n_particles
        # Sample the factored/joint declared priors directly; no enumeration of
        # their full Cartesian support and no access to an evaluation seed.
        capacities = np.zeros((N, len(s.legs)), dtype=np.int64)
        covered = set()
        for prior in s.capacity_priors:
            vectors = []
            prob = []
            for vector, p in zip(prior.support, prior.probabilities):
                if all(
                    first_obs.public_exo.get("capacity/" + leg, value) == value
                    for leg, value in zip(prior.legs, vector)
                ):
                    vectors.append(vector)
                    prob.append(p)
            if not prob:
                raise ZeroLikelihoodError(
                    "public joint capacity is outside prior support"
                )
            prob = np.asarray(prob) / sum(prob)
            draws = self.rng.choice(len(prob), size=N, p=prob)
            for k, leg in enumerate(prior.legs):
                capacities[:, c.leg_index[leg]] = np.array(vectors)[draws, k]
            covered.update(prior.legs)
        for l, leg in enumerate(s.legs):
            if leg.id in covered:
                continue
            values = []
            probs = []
            for value, p in zip(leg.capacity.support, leg.capacity.probabilities):
                if first_obs.public_exo.get("capacity/" + leg.id, value) == value:
                    values.append(value)
                    probs.append(p)
            if not probs:
                raise ZeroLikelihoodError(
                    "public leg capacity is outside prior support"
                )
            probs = np.asarray(probs) / sum(probs)
            capacities[:, l] = self.rng.choice(values, size=N, p=probs)
        capacities[:, list(c.legs_of[self.focal])] = first_obs.own_capacity
        bg = np.zeros((N, len(s.background)), dtype=int)
        if self.track_background:
            for k, b in enumerate(s.background):
                bg[:, k] = self.rng.choice(
                    b.capacity.support, size=N, p=b.capacity.probabilities
                )
        public = {
            k: v
            for k, v in first_obs.public_exo.items()
            if not k.startswith("capacity/")
        }
        valid = [
            k
            for k in range(len(s.exogenous.states))
            if public_signal(c, k) == public and s.exogenous.initial[k] > 0
        ]
        if not valid:
            raise ZeroLikelihoodError(
                "initial public exogenous signal has zero prior probability"
            )
        probs = np.array([s.exogenous.initial[k] for k in valid])
        probs /= probs.sum()
        exos = self.rng.choice(valid, size=N, p=probs)
        grouped = defaultdict(int)
        for cap, background, exo in zip(capacities, bg, exos):
            grouped[(tuple(cap), tuple(background), int(exo))] += 1
        mass = defaultdict(float)
        for (cap, background, exo), count in grouped.items():
            e = MarketEngine(c, check_invariants=False)
            e.reset(0)
            w = e.world
            w.exo = exo
            for i, cs in enumerate(w.carriers):
                cs.capacity = np.array([cap[l] for l in c.legs_of[i]])
            for state, stock in zip(w.background, background):
                state.stock = stock
            w.quote_history.initial = posted_quotes(w, c)
            ks = self.kernel.encode(w, self.focal)
            if not self._privileged_match(first_obs, ks):
                continue
            likelihood = sum(
                p
                for p, _ in self.kernel.observation_distribution(
                    ks, self.focal, self.arm, first_obs.quotes
                )
            )
            if likelihood:
                mass[self._hidden(ks)] += count / N * likelihood
        self._normalize(mass)
        self.particles = ExactBelief.sample(self, N, self.rng)
        self.weights = np.full(N, 1 / N)
        self._aggregate()
        self.last_observation = first_obs
        self.history = []
        self.diagnostics = []
        if self.rejuvenation_steps:
            self.reference = ExactBelief(
                c, s.carriers[self.focal].code, self.arm.name, self.kernel
            )
            self.reference.reset(first_obs)

    def update(self, before, action, after):
        """
        Condition the maintained posterior on the observed action/outcome sequence (Eqs.
        24–25, 30).
        """
        self._used_simulated = False
        groups = defaultdict(list)
        for k, (h, w) in enumerate(zip(self.particles, self.weights)):
            if w:
                groups[h].append(k)
        particles = list(self.particles)
        weights = np.zeros(self.n_particles)
        for h, indexes in groups.items():
            transitions = self.conditional_successors(h, before, action, after)
            likelihood = sum(transitions.values())
            if not likelihood:
                continue
            states, prob = zip(*sorted(transitions.items()))
            cdf = np.cumsum(prob) / likelihood
            indexes = np.asarray(indexes, dtype=int)
            if self.conditional_sampling == "systematic":
                # Randomly assign a systematic set to exchangeable proposals.
                # Every particle retains a uniform marginal proposal; shared
                # strata reduce avoidable conditional-transition variance.
                indexes = self.rng.permutation(indexes)
                uniforms = (self.rng.random() + np.arange(len(indexes))) / len(indexes)
            else:
                uniforms = self.rng.random(len(indexes))
            choices = np.minimum(
                np.searchsorted(cdf, uniforms, side="right"), len(states) - 1
            )
            for k, choice in zip(indexes, choices):
                particles[int(k)] = states[int(choice)]
            weights[indexes] = self.weights[indexes] * likelihood
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ZeroLikelihoodError(
                "particle update has zero weight; no truth substitution"
            )
        self.particles = particles
        self.weights = weights / total
        ess = float(1 / (self.weights @ self.weights))
        resampled = ess < self.ess_fraction * self.n_particles
        if resampled:
            self._resample()
        self._aggregate()
        self.last_observation = after
        self.history.append((before, action, after))
        self.diagnostics.append(
            {
                "period": after.t,
                "ess": ess,
                "resampled": resampled,
                "proposal": "conditional_simulation"
                if self._used_simulated
                else "exact_conditional",
                "conditional_sampling": self.conditional_sampling,
                "inner_draws": self.inner_draws if self._used_simulated else 0,
            }
        )
        if self.reference is not None:
            self.reference.update(before, action, after)
            self.rejuvenate(self.reference)

    def rejuvenate(self, reference):
        """
        Apply a finite-support posterior-preserving Metropolis–Hastings move (Eqs.
        24–25, 30).
        """
        # A finite-history posterior supplies the MH target on small instances.
        # A symmetric complete-graph independence proposal makes the density
        # ratio explicit and leaves the target invariant.
        if not np.allclose(self.weights, 1 / self.n_particles):
            self._resample()
        states = sorted(reference.mass)
        for _ in range(self.rejuvenation_steps):
            for k, h in enumerate(self.particles):
                candidate = states[int(self.rng.integers(len(states)))]
                old = reference.mass.get(h, 0.0)
                ratio = reference.mass[candidate] / old if old else math.inf
                if self.rng.random() < min(1.0, ratio):
                    self.particles[k] = candidate
        self._aggregate()

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 24–25, 30).
        """
        return (
            super().snapshot(),
            list(self.particles),
            self.weights.copy(),
            self.rng.bit_generator.state,
            list(self.diagnostics),
            self.reference.snapshot() if self.reference is not None else None,
        )

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 24–25, 30)."""
        super().restore(s[0])
        self.particles = list(s[1])
        self.weights = s[2].copy()
        self.rng.bit_generator.state = s[3]
        self.diagnostics = list(s[4])
        if s[5] is not None:
            self.reference.restore(s[5])
