"""Count-ledger exact transitions without MarketEngine allocation (Eq. 25).

The compact model certifies the inventory-driven/no-refund specialization.
The general ExactKernel remains the reference and handles other processes.
"""

from dataclasses import replace
from itertools import product
from collections import defaultdict
from ..core.types import CoefficientAction, PriceAction, HoldAction
from ..core.snapshot import FrozenMap
from ..pomdp.observation import Observation, PrivilegedState, QuoteObs
from .kernel import ExactKernel, FocalSignal, KernelState
from .finite import FinitePricingModel, UnsupportedFiniteModel
from .state import Booking
from .quotes import TrueQuote, quote_likelihood
from .background import quote as background_quote, purchase_probability
from .exogenous import public_signal


class FastExactKernel(ExactKernel):
    """API-compatible deterministic-quote Bernoulli specialization."""

    def __init__(self, compiled, focal, arm, controllers_spec=None, **kwargs):
        """
        Bind declared inputs and initialize FastExactKernel bookkeeping (Eqs. 3–16).
        """
        super().__init__(compiled, controllers_spec, **kwargs)
        self.model = FinitePricingModel(compiled, focal, arm, controllers_spec)
        self.focal = self.model.focal
        if any(
            q.fare_noise_sd or q.missing_probability or q.lag_periods
            for q in compiled.spec.quotes.requests
            if q.observer == compiled.spec.carriers[self.focal].code
        ):
            raise UnsupportedFiniteModel(
                "fast full-state kernel requires deterministic current focal shopping"
            )

    def _bookings(self, ks, carrier, redacted=False):
        """Materialize only immutable booking records needed by the observation."""
        c = self.compiled
        cs = ks.carriers[carrier]
        groups = []
        for local, j in enumerate(c.products_of[carrier]):
            fares = (
                cs.fares[local]
                if cs.fares is not None
                else (c.spec.products[j].reference_fare,) * cs.counts[local]
            )
            periods = cs.periods[local] if cs.periods else (-1,) * cs.counts[local]
            groups.append(iter(zip(fares, periods)))
        order = cs.booking_order or tuple(
            k for k, n in enumerate(cs.counts) for _ in range(n)
        )
        return tuple(
            Booking(k, carrier, c.products_of[carrier][local], fare, period, -1, -1, -1)
            for k, local in enumerate(order)
            for fare, period in [next(groups[local])]
        )

    def _quotes(self, ks, own_prices=None):
        """Current physical quotes using the same threshold and seat rules."""
        m = self.model
        c = self.compiled
        own = ks.carriers[self.focal]
        prices = own_prices or own.posted_prices or own.previous_prices
        inventories = [m.inventory(i, cs) for i, cs in enumerate(ks.carriers)]
        rival = m.rival_quotes(ks.t, inventories[m.rival], own.previous_prices)
        quotes = []
        for j, p in enumerate(c.spec.products):
            i = int(c.product_carrier[j])
            local = c.products_of[i].index(j)
            if i == self.focal:
                use = m.own_uses[local]
                seats = min(x for x, u in zip(inventories[i], use) if u)
                fare = prices[local]
            else:
                fare, _, seats = rival[local]
            quotes.append(TrueQuote(j, ks.t, float(fare), seats > 0, max(0, seats)))
        for k, (stock, spec) in enumerate(zip(ks.background, c.spec.background)):
            fare, available, seats = background_quote(spec, stock)
            quotes.append(
                TrueQuote(len(c.spec.products) + k, ks.t, fare, available, seats)
            )
        return tuple(quotes)

    def observation(self, ks, arm=None, purchases=()):
        """Construct scoped actor data, without random tapes or hidden references."""
        m = self.model
        c = self.compiled
        arm = arm or m.arm
        cs = ks.carriers[self.focal]
        truth = self._quotes(ks)
        requests = {q.id: q for q in c.spec.quotes.requests}
        names = {
            **c.product_index,
            **{b.id: len(c.spec.products) + i for i, b in enumerate(c.spec.background)},
        }
        quotes = []
        for qid in arm.requests:
            request = requests[qid]
            if ks.t % request.every_k_periods:
                continue
            for target in request.target_products:
                j = names[target]
                q = truth[j]
                fare = (
                    round(round(q.fare / request.fare_bin) * request.fare_bin, 2)
                    if request.fare_bin
                    else q.fare
                )
                quotes.append(
                    QuoteObs(
                        j,
                        ks.t,
                        fare,
                        q.available if request.availability == "flag" else None,
                        min(q.seats, request.seats_cap)
                        if request.availability == "seats_capped"
                        else None,
                        False,
                    )
                )
        public = dict(public_signal(c, ks.exo))
        joint_public = {
            leg
            for prior in c.spec.capacity_priors
            if prior.public
            for leg in prior.legs
        }
        for l, leg in enumerate(c.spec.legs):
            if leg.capacity.public or leg.id in joint_public:
                owner = int(c.leg_carrier[l])
                public["capacity/" + leg.id] = float(
                    ks.carriers[owner].capacity[c.legs_of[owner].index(l)]
                )
        privileged = (
            PrivilegedState(
                tuple(x.capacity for x in ks.carriers),
                tuple(self._bookings(ks, i) for i in range(len(ks.carriers))),
                tuple(x.controller_memory for x in ks.carriers),
                ks.exo,
            )
            if arm.privileged
            else None
        )
        return Observation(
            ks.t,
            self.focal,
            arm.name,
            cs.capacity,
            self._bookings(ks, self.focal, True),
            cs.cash,
            cs.refunds,
            cs.previous_prices,
            cs.posted_prices,
            FrozenMap(public),
            tuple(quotes),
            m.inventory(self.focal, cs),
            privileged,
            purchases,
        )

    def observation_distribution(self, ks, focal, arm, observed_quotes=None):
        """Deterministic likelihood; unsupported carriers use the reference API."""
        if focal != self.focal:
            return super().observation_distribution(ks, focal, arm, observed_quotes)
        obs = self.observation(ks, arm)
        return (
            [(1.0, obs)]
            if observed_quotes is None or tuple(observed_quotes) == obs.quotes
            else []
        )

    def transition(
        self,
        ks,
        focal,
        focal_action,
        *,
        arm=None,
        observation=None,
        observed_quotes=None,
    ):
        """Enumerate product outcomes, integrating request type and background."""
        if focal != self.focal:
            return super().transition(
                ks,
                focal,
                focal_action,
                arm=arm,
                observation=observation,
                observed_quotes=observed_quotes,
            )
        m = self.model
        c = self.compiled
        arm = arm or m.arm
        own = ks.carriers[focal]
        obs = observation or self.observation(ks, arm)
        if isinstance(focal_action, HoldAction):
            focal_action = CoefficientAction(own.theta)
        if isinstance(focal_action, CoefficientAction):
            theta = focal_action.theta
            prices = tuple(map(float, m.rule.prices(theta, obs)))
        elif isinstance(focal_action, PriceAction):
            theta = own.theta
            prices = tuple(float(focal_action.prices[j]) for j in c.products_of[focal])
        else:
            return super().transition(
                ks,
                focal,
                focal_action,
                arm=arm,
                observation=observation,
                observed_quotes=observed_quotes,
            )
        own_stock = m.inventory(focal, own)
        hidden = m.hidden(ks)
        rival_quotes = m.rival_quotes(ks.t, hidden[0], own.previous_prices)
        history = dict(ks.history)
        history[ks.t] = self._quotes(ks, prices)
        lag = max([q.lag_periods for q in c.spec.quotes.requests] or [1])
        history = tuple(
            (t, row) for t, row in sorted(history.items()) if t >= ks.t + 1 - lag
        )
        bg_options = []
        for stock, spec in zip(ks.background, c.spec.background):
            p = purchase_probability(spec, stock)
            bg_options.append([(1 - p, stock), (p, max(0, stock - 1))])
        result = []
        for chosen, p in enumerate(
            m.outcomes(ks.t, own_stock, hidden[0], prices, own.previous_prices)
        ):
            if not p:
                continue
            carriers = []
            purchases = ()
            cash_delta = 0.0
            for i, source in enumerate(ks.carriers):
                counts = list(source.counts)
                fares = source.fares
                periods = source.periods
                order = source.booking_order
                cash = source.cash
                if (
                    chosen < len(c.spec.products)
                    and int(c.product_carrier[chosen]) == i
                ):
                    local = c.products_of[i].index(chosen)
                    counts[local] += 1
                    if i == focal:
                        fare = prices[local]
                        cash_delta = fare
                        cash += fare
                        purchases = ((chosen, fare),)
                        fares = tuple(
                            (*group, fare) if k == local else group
                            for k, group in enumerate(fares)
                        )
                        periods = tuple(
                            (*group, ks.t) if k == local else group
                            for k, group in enumerate(periods)
                        )
                        order = (*order, local)
                if i != focal:
                    cash = sum(
                        n * c.spec.products[j].reference_fare
                        for n, j in zip(counts, c.products_of[i])
                    )
                    order = tuple(k for k, n in enumerate(counts) for _ in range(n))
                    periods = tuple((-1,) * n for n in counts)
                posted = prices if i == focal else tuple(q[0] for q in rival_quotes)
                carriers.append(
                    replace(
                        source,
                        counts=tuple(counts),
                        fares=fares,
                        periods=periods,
                        booking_order=order,
                        cash=cash,
                        theta=theta if i == focal else source.theta,
                        previous_prices=posted,
                        posted_prices=posted,
                    )
                )
            cash_delta = carriers[focal].cash - own.cash
            for bg in product(*bg_options):
                probability = p
                for w, _ in bg:
                    probability *= w
                if not probability:
                    continue
                state = KernelState(
                    ks.t + 1,
                    m.exos[ks.t + 1],
                    tuple(carriers),
                    tuple(stock for _, stock in bg),
                    "terminal" if ks.t + 1 >= m.T else "pre_decision",
                    history,
                    ks.initial_quotes,
                    ks.events_fired,
                    (),
                )
                state = self.canonical_state(state, focal)
                after = self.observation(state, arm, purchases)
                if (
                    observed_quotes is not None
                    and tuple(observed_quotes) != after.quotes
                ):
                    continue
                signal = FocalSignal(
                    purchases,
                    after.own_capacity,
                    tuple(sorted(after.public_exo.items())),
                    cash_delta,
                    after,
                )
                result.append((probability, state, signal))
        return result


def make_exact_kernel(compiled, focal, arm):
    """Select a verified specialization or the unchanged general exact kernel."""
    try:
        return FastExactKernel(compiled, focal, arm)
    except UnsupportedFiniteModel:
        return ExactKernel(compiled)
