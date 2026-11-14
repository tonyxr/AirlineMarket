"""Exact one-period rival-offer memory for the finite Bernoulli specialization.

The hidden state retains current stock/journeys and the last posted offers
queried with lag one. A lagged signal never substitutes today's inventory.
The original finite model's other applicability conditions are unchanged.
"""

from dataclasses import replace

from ..agents.price_rule import AdditivePriceRule
from .finite import FinitePricingModel, UnsupportedFiniteModel


class LaggedFinitePricingModel(FinitePricingModel):
    """Augment physical state with only the rival quote rows needed next period."""

    def __init__(self, compiled, focal, arm, rival_controllers=None):
        """Reuse physical-model certification, then restore the declared quote channel."""
        code = focal if isinstance(focal, str) else compiled.spec.carriers[focal].code
        selected = (next(a for a in compiled.spec.information if a.observer == code and a.name == arm)
                    if isinstance(arm, str) else arm)
        focal_requests = [q for q in compiled.spec.quotes.requests if q.id in selected.requests]
        if any(q.lag_periods not in (0, 1) for q in focal_requests):
            raise UnsupportedFiniteModel("finite history revision supports focal lag zero or one only")
        # This private construction view bypasses only the old lag refusal.
        # No transition, observation or result is evaluated under this view.
        shadow_spec = replace(compiled.spec, quotes=replace(
            compiled.spec.quotes, requests=tuple(
                replace(q, lag_periods=0) if q.observer == code else q
                for q in compiled.spec.quotes.requests)))
        super().__init__(replace(compiled, spec=shadow_spec), focal, arm, rival_controllers)
        self.compiled = compiled
        self.spec = compiled.spec
        requests = {q.id: q for q in self.spec.quotes.requests}
        self.requests = tuple(requests[q] for q in self.arm.requests
                              if any(p in compiled.product_index for p in requests[q].target_products))
        if any(compiled.product_carrier[compiled.product_index[p]] != self.rival
               for q in self.requests for p in q.target_products if p in compiled.product_index):
            raise UnsupportedFiniteModel("finite focal shopping must target rival products")
        self.history_targets = {compiled.product_index[p] for q in self.requests
                                if q.lag_periods for p in q.target_products
                                if p in compiled.product_index}
        self.has_lag = bool(self.history_targets)
        self.rule = AdditivePriceRule(compiled, self.pricing, self.arm)
        self._stock_only_last_step = self._all_competing_offers_visible()
        response_products = {compiled.product_index[p] for p, rule in self.rival_rules.items()
                             if rule.get("responses")}
        observed_current_fares = {compiled.product_index[p] for q in self.requests
                                  if q.lag_periods == 0 and q.every_k_periods == 1
                                  and q.fare_bin == 0 for p in q.target_products
                                  if p in compiled.product_index}
        self.previous_encoded_by_signal = response_products <= observed_current_fares

    def _all_competing_offers_visible(self):
        """Permit stock-only terminal reduction only for visible CURRENT competition."""
        needed = {j for j in self.rival_products if self.spec.products[j].market
                  in {self.spec.products[k].market for k in self.own_products}}
        observed = {self.compiled.product_index[p] for q in self.requests
                    if q.lag_periods == 0 and q.every_k_periods == 1
                    and q.availability in ("flag", "seats_capped") and q.fare_bin == 0
                    for p in q.target_products if p in self.compiled.product_index}
        return needed <= observed and not self.track_journeys

    def _remember(self, quotes):
        """Canonicalize unqueried offers while retaining queried true fare/stock data."""
        return tuple(quote if j in self.history_targets else (0.0, False, 0)
                     for j, quote in zip(self.rival_products, quotes))

    def hidden(self, kernel_state):
        """Read model-generated last-posted history, including initial negative time."""
        physical = super().hidden(kernel_state)
        if not self.has_lag:
            return physical
        period = kernel_state.t - 1
        if period < 0:
            row = kernel_state.initial_quotes
        else:
            row = dict(kernel_state.history)[period]
        remembered = self._remember(tuple((row[j].fare, row[j].available, row[j].seats)
                                           for j in self.rival_products))
        return (*physical, remembered)

    def step(self, t, own, own_journeys, hidden, prices, previous):
        """Retain PRE-purchase posted offers when advancing the physical stock law."""
        remembered = self._remember(self.rival_quotes(t, hidden[0], previous)) if self.has_lag else None
        for p, new_own, journeys, new_hidden, reward, sale in super().step(
                t, own, own_journeys, hidden[:2], prices, previous):
            state = (*new_hidden, remembered) if self.has_lag else new_hidden
            yield p, new_own, journeys, state, reward, sale

    def signal(self, t, hidden, previous):
        """Apply each request's true lag, cadence, bin and availability channel."""
        current = self.rival_quotes(t, hidden[0], previous)
        result = []
        for request in self.requests:
            if t % request.every_k_periods:
                continue
            source = hidden[2] if request.lag_periods else current
            for target in request.target_products:
                if target not in self.compiled.product_index:
                    continue
                j = self.compiled.product_index[target]
                fare, available, seats = source[self._rival_local[j]]
                if request.fare_bin:
                    fare = round(round(fare / request.fare_bin) * request.fare_bin, 2)
                result.append((j, fare, available if request.availability == "flag" else None,
                               min(seats, request.seats_cap)
                               if request.availability == "seats_capped" else None))
        return tuple(result), hidden[:2] if self.arm.privileged else None
