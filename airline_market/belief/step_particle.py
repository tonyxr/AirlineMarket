"""Original strict particle proposal with an explicit exact one-step tape revision."""

from collections import defaultdict

from airline_market.belief.exact import ExactBelief
from airline_market.belief.template_particle import TemplateParticleBelief
from airline_market.core.errors import KernelLimitError


class StepTapeParticleBelief(TemplateParticleBelief):
    """Preserve proposal logic and RNG; change only unused tape materialization."""

    def conditional_successors(self, h, before, action, after):
        """
        Enumerate next hidden states consistent with the observed own outcome (Eqs.
        24–25, 30).
        """
        if self.exact_proposals:
            try:
                return ExactBelief.conditional_successors(
                    self, h, before, action, after
                )
            except KernelLimitError:
                pass
        # Conditional simulation estimates the joint focal-signal/quote mass.
        # Only fresh planning draws are used; continuous quote likelihoods are
        # integrated analytically rather than requiring equal noisy fares.
        from ..core.types import CoefficientAction, PriceAction
        from ..sim.step_tape import OneStepTape
        from .exact import ledger_key

        result = defaultdict(float)
        for _ in range(self.inner_draws):
            e = self.kernel.engine(h.kernel, self.focal)
            e.tape = OneStepTape.generate(
                self.compiled, int(self.rng.integers(2**63)), period=before.t
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
            step = e.step({self.focal: effective})  # noqa: F841 - original proposal body
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
