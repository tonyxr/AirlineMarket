"""Compact exact Bernoulli network model (Eqs. 3, 7–9, 18, 25–27).

Only inventory-driven, deterministic quoted-offer models are reduced here.
Capacity events, latent exogenous transitions and richer controllers continue
through ExactKernel. No probability pruning or belief rounding is performed.
"""

from collections import defaultdict
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from ..agents.price_rule import AdditivePriceRule
from ..sim.arrivals import bernoulli_probability
from ..sim.choice import systematic_utilities, choice_probabilities
from ..sim.exogenous import transition_row, public_signal
from ..sim.tape import segment_weights
from ..pomdp.observation import QuoteObs


class UnsupportedFiniteModel(ValueError):
    """The specified process needs the general exact kernel."""


class FinitePricingModel:
    """Sufficient inventory/journey states for a fixed-capacity duopoly.

    Carrier stock and segment journeys are Markov sufficient when coefficients
    use inventory/public/current-quote features and rivals use threshold rules.
    Journey counts are marginalized when share has zero objective weight.
    Independent background quotes are ancillary and integrate to one.
    """

    def __init__(self, compiled, focal, arm, rival_controllers=None):
        """
        Bind declared inputs and initialize FinitePricingModel bookkeeping (Eqs. 3–16).
        """
        self.compiled = compiled
        self.spec = s = compiled.spec
        self.focal = compiled.carrier_index[focal] if isinstance(focal, str) else focal
        self.rival = 1 - self.focal
        self.arm = (
            next(
                a
                for a in s.information
                if a.observer == s.carriers[self.focal].code and a.name == arm
            )
            if isinstance(arm, str)
            else arm
        )
        self.T = s.horizon.n_periods
        self.own_products = compiled.products_of[self.focal]
        self.rival_products = (
            compiled.products_of[self.rival] if len(s.carriers) == 2 else ()
        )
        self.pricing = next(
            p for p in s.pricing if p.carrier == s.carriers[self.focal].code
        )
        self.objective = next(
            o for o in s.objectives if o.carrier == s.carriers[self.focal].code
        )
        self.track_journeys = self.objective.share_weight != 0
        self.markets = set(
            self.objective.share_markets or [m.id for m in s.markets if m.in_segment]
        )
        self.rule = AdditivePriceRule(compiled, self.pricing, self.arm)
        if (
            len(s.carriers) != 2
            or s.demand.mode != "single_request_bernoulli"
            or s.choice.model != "latent_class_mnl"
        ):
            raise UnsupportedFiniteModel(
                "compact exact model requires a Bernoulli latent-class duopoly"
            )
        if s.timing.mode != "leader_follower" or s.timing.leaders != (
            s.carriers[self.rival].code,
        ):
            raise UnsupportedFiniteModel(
                "compact model requires the inventory-driven rival to post first"
            )
        if self.pricing.update_every != 1:
            raise UnsupportedFiniteModel(
                "held coefficients require an additional action-memory state"
            )
        supported = {
            "intercept",
            "product_indicator",
            "connection_indicator",
            "scarcity_sum",
            "scarcity_max",
            "time_to_departure",
            "direct_rival_gap",
            "min_rival_gap",
            "rival_availability",
            "public_disruption",
            "linked_scarcity_signal",
        }
        if any(
            f.name not in supported
            for f in (
                *self.pricing.controllable_features,
                *(f for f, _ in self.pricing.fixed_features),
            )
        ):
            raise UnsupportedFiniteModel(
                "the coefficient basis needs more than inventory and current quotes"
            )
        initial = [k for k, p in enumerate(s.exogenous.initial) if p > 0]
        if len(initial) != 1:
            raise UnsupportedFiniteModel(
                "initial exogenous uncertainty needs the general kernel"
            )
        exos = [initial[0]]
        for t in range(self.T):
            row = transition_row(compiled, exos[-1], t)
            targets = [k for k, p in enumerate(row) if p > 0]
            if len(targets) != 1 or row[targets[0]] != 1.0:
                raise UnsupportedFiniteModel(
                    "stochastic exogenous transitions need the general kernel"
                )
            exos.append(targets[0])
        self.exos = tuple(exos)
        for event in s.capacity_process.events:
            transitions = event.trigger.get("exo_transition")
            if not transitions or any(
                [s.exogenous.states[a].name, s.exogenous.states[b].name]
                == list(transitions)
                and a != b
                for a, b in zip(exos, exos[1:])
            ):
                raise UnsupportedFiniteModel(
                    "active capacity events require fare ledgers and resolution"
                )
        assignment = next(
            a for a in s.controllers if a.carrier == s.carriers[self.rival].code
        )
        override = (rival_controllers or {}).get(
            self.rival, (rival_controllers or {}).get(assignment.carrier)
        )
        if override is not None:
            from ..agents.rule import ThresholdLinearRuleController

            if not isinstance(override, ThresholdLinearRuleController):
                raise UnsupportedFiniteModel(
                    "controller override is not an inventory threshold rule"
                )
            self.rival_rules = override.rules
        elif assignment.controller == "threshold_linear_rule":
            self.rival_rules = assignment.params.get("rules", {})
        else:
            raise UnsupportedFiniteModel(
                "rival controller requires the general exact kernel"
            )
        requests = {q.id: q for q in s.quotes.requests}
        self.requests = tuple(
            requests[q]
            for q in self.arm.requests
            if any(p in compiled.product_index for p in requests[q].target_products)
        )
        if any(
            q.lag_periods or q.fare_noise_sd or q.missing_probability
            for q in self.requests
        ):
            raise UnsupportedFiniteModel(
                "stale or stochastic focal quotes need observation-history state"
            )
        for q in s.quotes.requests:
            if q.observer == assignment.carrier and (
                q.lag_periods != 1
                or q.fare_noise_sd
                or q.missing_probability
                or q.fare_bin
                or q.every_k_periods != 1
            ):
                raise UnsupportedFiniteModel(
                    "rival responses require noiseless previous focal fares"
                )
        for rule in self.rival_rules.values():
            if any(
                compiled.product_index[r["product"]] not in self.own_products
                for r in rule.get("responses", ())
            ):
                raise UnsupportedFiniteModel(
                    "rival response uses an undeclared price history"
                )
        self.own_uses = tuple(
            tuple(int(x) for x in compiled.A[self.focal][:, j])
            for j in range(len(self.own_products))
        )
        self.rival_uses = tuple(
            tuple(int(x) for x in compiled.A[self.rival][:, j])
            for j in range(len(self.rival_products))
        )
        self.own_costs = tuple(
            float(compiled.product_cost[j]) for j in self.own_products
        )
        self._rival_local = {j: k for k, j in enumerate(self.rival_products)}
        self._own_local = {j: k for k, j in enumerate(self.own_products)}
        self._stock_only_last_step = self._all_competing_offers_visible()
        observed_fares = {
            compiled.product_index[p]
            for q in self.requests
            if q.every_k_periods == 1 and q.fare_bin == 0
            for p in q.target_products
            if p in compiled.product_index
        }
        response_products = {
            compiled.product_index[p]
            for p, rule in self.rival_rules.items()
            if rule.get("responses")
        }
        self.previous_encoded_by_signal = response_products <= observed_fares
        self.cross_market_features = any(
            f.name == "linked_scarcity_signal"
            or (
                f.name == "direct_rival_gap"
                and any(
                    s.products[compiled.product_index[source]].market
                    != s.products[compiled.product_index[target]].market
                    for source, target in f.params.get("targets", {}).items()
                )
            )
            for f in (
                *self.pricing.controllable_features,
                *(f for f, _ in self.pricing.fixed_features),
            )
        )
        # Per-instance caches must not keep discarded market objects alive.
        self.rival_quotes = lru_cache(maxsize=100000)(self.rival_quotes)
        self.signal = lru_cache(maxsize=100000)(self.signal)
        self.prices = lru_cache(maxsize=250000)(self.prices)
        self._outcomes = lru_cache(maxsize=200000)(self._outcomes)

    def _all_competing_offers_visible(self):
        """Whether last-period profit is determined by observed offers alone."""
        needed = {
            j
            for j in self.rival_products
            if self.spec.products[j].market
            in {self.spec.products[k].market for k in self.own_products}
        }
        observed = {
            self.compiled.product_index[p]
            for q in self.requests
            if q.every_k_periods == 1
            and q.availability in ("flag", "seats_capped")
            and q.fare_bin == 0
            for p in q.target_products
            if p in self.compiled.product_index
        }
        return needed <= observed and not self.track_journeys

    def inventory(self, carrier, state):
        """Recover leg stocks from a count ledger (Eq. 3)."""
        return tuple(
            int(k - sum(int(a) * n for a, n in zip(row, state.counts)))
            for k, row in zip(state.capacity, self.compiled.A[carrier])
        )

    def journeys(self, carrier, state):
        """Count competitive journeys only when terminal share needs them."""
        return (
            sum(
                n
                for j, n in zip(self.compiled.products_of[carrier], state.counts)
                if self.spec.products[j].market in self.markets
            )
            if self.track_journeys
            else 0
        )

    def hidden(self, kernel_state):
        """Project a full rival ledger onto its decision-relevant state."""
        r = kernel_state.carriers[self.rival]
        return self.inventory(self.rival, r), self.journeys(self.rival, r)

    def rival_quotes(self, t, stock, previous):
        """Recompute endogenous rival offers from inventory and previous fares."""
        c = self.compiled
        prices = []
        for local, j in enumerate(self.rival_products):
            product = self.spec.products[j]
            rule = self.rival_rules.get(product.id, {"base": product.reference_fare})
            fare = rule.get("base", product.reference_fare)
            for term in rule.get("scarcity", ()):
                leg = c.legs_of[self.rival].index(c.leg_index[term["leg"]])
                fare += term["coefficient"] * (stock[leg] <= term["threshold"])
            for term in rule.get("responses", ()):
                fare += term["coefficient"] * (
                    previous[self._own_local[c.product_index[term["product"]]]]
                    - term["reference"]
                )
            seats = min(x for x, u in zip(stock, self.rival_uses[local]) if u)
            prices.append((round(float(fare), 2), seats > 0, max(0, seats)))
        return tuple(prices)

    def signal(self, t, hidden, previous):
        """Return the current rival shopping signal, marginalizing background."""
        quotes = self.rival_quotes(t, hidden[0], previous)
        result = []
        for request in self.requests:
            if t % request.every_k_periods:
                continue
            for target in request.target_products:
                if target not in self.compiled.product_index:
                    continue
                j = self.compiled.product_index[target]
                fare, available, seats = quotes[self._rival_local[j]]
                fare = (
                    round(round(fare / request.fare_bin) * request.fare_bin, 2)
                    if request.fare_bin
                    else fare
                )
                result.append(
                    (
                        j,
                        fare,
                        available if request.availability == "flag" else None,
                        min(seats, request.seats_cap)
                        if request.availability == "seats_capped"
                        else None,
                    )
                )
        return tuple(result), hidden if self.arm.privileged else None

    def observed_signal(self, obs):
        """Extract the same signal from a permitted actor observation."""
        quotes = tuple(
            (q.product, q.fare, q.available, q.seats)
            for q in obs.quotes
            if q.product < len(self.spec.products)
        )
        hidden = None
        if self.arm.privileged:
            truth = obs.require_privileged()
            counts = tuple(
                sum(b.product == j for b in truth.bookings[self.rival])
                for j in self.rival_products
            )
            stock = tuple(
                int(k - sum(int(a) * n for a, n in zip(row, counts)))
                for k, row in zip(
                    truth.capacities[self.rival], self.compiled.A[self.rival]
                )
            )
            count = (
                sum(
                    n
                    for j, n in zip(self.rival_products, counts)
                    if self.spec.products[j].market in self.markets
                )
                if self.track_journeys
                else 0
            )
            hidden = stock, count
        return quotes, hidden

    def prices(self, theta, t, capacity, own, signal):
        """Use the shared additive price rule, including Python cent rounding."""
        obs = SimpleNamespace(
            t=t,
            own_capacity=capacity,
            own_inventory=own,
            own_bookings=(),
            public_exo=public_signal(self.compiled, self.exos[t]),
            quotes=tuple(
                QuoteObs(j, t, fare, available, seats, False)
                for j, fare, available, seats in signal[0]
            ),
        )
        return tuple(map(float, self.rule.prices(theta, obs)))

    def outcomes(self, t, own, rival, prices, previous):
        """Exact probabilities of each product purchase and no purchase."""
        rival_prices = tuple(q[0] for q in self.rival_quotes(t, rival, previous))
        return self._outcomes(t, own, rival, prices, rival_prices)

    def _outcomes(self, t, own, rival, prices, rival_prices):
        """Cache by actual offers rather than finer, irrelevant past cents."""
        c = self.compiled
        global_prices = np.zeros(len(self.spec.products))
        global_prices[list(self.own_products)] = prices
        global_prices[list(self.rival_products)] = rival_prices
        active = np.zeros(len(global_prices), dtype=bool)
        active[list(self.own_products)] = [
            all(x >= u for x, u in zip(own, use)) for use in self.own_uses
        ]
        active[list(self.rival_products)] = [
            all(x >= u for x, u in zip(rival, use)) for use in self.rival_uses
        ]
        arrival = bernoulli_probability(c, t, self.exos[t])
        probabilities = np.zeros(len(global_prices) + 1)
        probabilities[-1] = 1 - arrival
        for market, weight in enumerate(self.spec.demand.market_weights.value_at(t)):
            for segment, mix in enumerate(segment_weights(c, t, market)):
                for taste in self.spec.segments[segment].classes:
                    p = choice_probabilities(
                        systematic_utilities(c, market, taste, global_prices, active)
                    )
                    mass = arrival * weight * mix * taste.weight
                    for local, j in enumerate(c.market_products[market]):
                        probabilities[j] += mass * p[local]
                    probabilities[-1] += mass * p[-1]
        return tuple(map(float, probabilities))

    def step(self, t, own, own_journeys, hidden, prices, previous):
        """Advance one request without constructing a WorldState (Eq. 25)."""
        for j, p in enumerate(self.outcomes(t, own, hidden[0], prices, previous)):
            if not p:
                continue
            next_own, next_journeys, next_hidden = own, own_journeys, hidden
            reward, sale = 0.0, -1
            if j in self._own_local:
                sale = self._own_local[j]
                next_own = tuple(x - u for x, u in zip(own, self.own_uses[sale]))
                next_journeys += int(
                    self.track_journeys and self.spec.products[j].market in self.markets
                )
                reward = (
                    prices[sale] - self.own_costs[sale]
                ) / self.objective.profit_scale
            elif j in self._rival_local:
                local = self._rival_local[j]
                next_hidden = (
                    tuple(x - u for x, u in zip(hidden[0], self.rival_uses[local])),
                    hidden[1]
                    + int(
                        self.track_journeys
                        and self.spec.products[j].market in self.markets
                    ),
                )
            yield p, next_own, next_journeys, next_hidden, reward, sale

    def terminal(self, own_journeys, hidden):
        """Terminal share after per-booking service costs have been settled."""
        total = own_journeys + hidden[1]
        return self.objective.share_weight * own_journeys / total if total else 0.0

    def sold_out(self, own):
        """Detect zero future focal revenue without enumerating rival sales."""
        return not any(all(x >= u for x, u in zip(own, use)) for use in self.own_uses)

    def monopoly(self, own, belief):
        """Whether rival competitive supply is irreversibly exhausted."""
        if not self._stock_only_last_step or self.cross_market_features:
            return False
        markets = {
            self.spec.products[j].market
            for j, use in zip(self.own_products, self.own_uses)
            if all(x >= u for x, u in zip(own, use))
        }
        rivals = [
            use
            for j, use in zip(self.rival_products, self.rival_uses)
            if self.spec.products[j].market in markets
        ]
        return all(
            not any(all(x >= u for x, u in zip(hidden[0], use)) for use in rivals)
            for hidden, _ in belief
        )

    def immediate_signal(self, own, signal):
        """Current profit ignores unavailable fares and irrelevant seat counts."""
        markets = {
            self.spec.products[j].market
            for j, use in zip(self.own_products, self.own_uses)
            if all(x >= u for x, u in zip(own, use))
        }
        result = []
        for j, fare, available, seats in signal[0]:
            if (
                not self.cross_market_features
                and self.spec.products[j].market not in markets
            ):
                continue
            visible = available is not False and (seats is None or seats > 0)
            result.append((j, fare if visible else 0.0, visible))
        return tuple(result)

    def project(self, obs, belief):
        """Build a normalized inventory posterior with no cash or booking dates."""
        mass = defaultdict(float)
        for h, p in belief.mass.items():
            mass[self.hidden(h.kernel)] += p
        own_journeys = (
            sum(
                self.spec.products[b.product].market in self.markets
                for b in obs.own_bookings
            )
            if self.track_journeys
            else 0
        )
        return (
            obs.t,
            obs.own_capacity,
            obs.own_inventory,
            own_journeys,
            obs.own_previous_prices,
            self.observed_signal(obs),
            tuple(sorted(mass.items())),
        )

    def existing_service_cost(self, obs):
        """Known sunk service commitment, excluded from new-booking rewards."""
        return (
            sum(float(self.compiled.product_cost[b.product]) for b in obs.own_bookings)
            / self.objective.profit_scale
        )
