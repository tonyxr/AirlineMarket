"""Observation-measurable additive price basis (Eqs. 18–19)."""

import numpy as np
from ..core.errors import InformationLeakError
from ..core.money import round_fare, clip_fare
from ..spec.features import FEATURES


def visible(q):
    """
    Test whether a shopping result is present and shows a sellable offer (Eqs. 18–19).
    """
    return (
        q is not None
        and not q.missing
        and q.fare is not None
        and (q.available is not False)
        and (q.seats is None or q.seats > 0)
    )


class FeatureEvaluator:
    """
    Map permitted carrier observations to declared scalar pricing features (Eqs. 18–19).
    """

    def __init__(self, compiled, carrier: int):
        """
        Bind declared inputs and initialize FeatureEvaluator bookkeeping (Eqs. 18–19).
        """
        self.c = compiled
        self.i = carrier

    def __call__(self, feature, j: int, obs):
        """Evaluate one named observable feature for one own itinerary (Eqs. 18–19)."""
        c = self.c
        p = c.spec.products[j]
        name = feature.name
        params = feature.params
        local = c.products_of[self.i].index(j)
        uses = c.A[self.i][:, local]
        if name == "intercept":
            return 1.0
        if name == "product_indicator":
            return float(p.id in params.get("products", ()))
        if name == "connection_indicator":
            return float(len(p.legs) > 1)
        if name == "scarcity_sum":
            return sum(float(u) / (1 + x) for u, x in zip(uses, obs.own_inventory))
        if name == "scarcity_max":
            return max(
                (
                    u * k / (1 + x)
                    for u, k, x in zip(uses, obs.own_capacity, obs.own_inventory)
                ),
                default=0.0,
            )
        if name == "time_to_departure":
            return (c.spec.horizon.n_periods - obs.t) / c.spec.horizon.n_periods
        if name == "booking_pace":
            expected = params.get("expected", {}).get(p.market, 0.0)
            if isinstance(expected, (tuple, list)):
                expected = expected[min(obs.t, len(expected) - 1)]
            actual = sum(
                c.product_market[b.product] == c.product_market[j]
                for b in obs.own_bookings
            )
            return (actual - expected) / max(1.0, expected)
        if getattr(self, "_observation", None) is not obs:
            self._observation = obs
            self._quotes = {q.product: q for q in obs.quotes}
            self._by_market = {}
            for q in obs.quotes:
                if q.product < len(c.spec.products):
                    self._by_market.setdefault(
                        int(c.product_market[q.product]), []
                    ).append(q)
        quotes = self._by_market.get(int(c.product_market[j]), ())
        if name == "direct_rival_gap":
            target = params.get("targets", {}).get(p.id)
            q = (
                self._quotes.get(c.product_index[target])
                if target is not None
                else next(iter(quotes), None)
            )
            return (
                q.fare - c.spec.products[q.product].reference_fare
                if visible(q)
                else params.get("missing_value", 0.0)
            )
        if name == "min_rival_gap":
            return min(
                (q.fare - p.reference_fare for q in quotes if visible(q)),
                default=params.get("missing_value", 0.0),
            )
        if name == "rival_availability":
            return float(any(visible(q) for q in quotes))
        if name == "public_disruption":
            return obs.public_exo.get("disrupted", 0.0)
        if name == "linked_scarcity_signal":
            targets = {c.product_index[x] for x in params.get("products", ())}
            return float(
                any(
                    q.product in targets
                    and visible(q)
                    and q.fare >= params.get("threshold", 0.0)
                    for q in obs.quotes
                )
            )
        if name == "belief_joint_availability":
            raise InformationLeakError(
                "belief feature requires an explicitly labeled belief policy adapter"
            )
        raise ValueError(f"unknown feature {name}")


class AdditivePriceRule:
    """Shared additive, bounded and cent-rounded itinerary pricing map (Eqs. 18–19)."""

    def __init__(self, compiled, pricing_spec, arm=None):
        """
        Bind declared inputs and initialize AdditivePriceRule bookkeeping (Eqs. 18–19).
        """
        self.c = compiled
        self.spec = pricing_spec
        self.i = compiled.carrier_index[pricing_spec.carrier]
        self.evaluator = FeatureEvaluator(compiled, self.i)
        if arm is not None:
            requests = {q.id: q for q in compiled.spec.quotes.requests}
            targets = {p for qid in arm.requests for p in requests[qid].target_products}
            for f in (
                *[x[0] for x in pricing_spec.fixed_features],
                *pricing_spec.controllable_features,
            ):
                if f.name not in FEATURES:
                    raise ValueError(f"unknown feature {f.name}")
                needs = FEATURES[f.name].needs
                if needs in ("direct", "linked"):
                    required = (
                        set(f.params.get("targets", {}).values())
                        if needs == "direct"
                        else set(f.params.get("products", ()))
                    )
                    if required - targets:
                        raise InformationLeakError(
                            f"{f.name} requires quotes {sorted(required - targets)}"
                        )
                    if needs == "direct" and not required:
                        for j in compiled.products_of[self.i]:
                            rival = {
                                compiled.spec.products[k].id
                                for k in compiled.market_products[
                                    int(compiled.product_market[j])
                                ]
                                if compiled.product_carrier[k] != self.i
                            }
                            if rival and not rival.intersection(targets):
                                raise InformationLeakError(
                                    f"{f.name} lacks a direct-market shopping request"
                                )

    def feature_matrix(self, obs):
        """
        Evaluate fixed and controllable features for each own product (Eqs. 18–19).
        """
        prods = self.c.products_of[self.i]
        fixed = np.array(
            [
                [self.evaluator(f, j, obs) for f, _ in self.spec.fixed_features]
                for j in prods
            ]
        )
        control = np.array(
            [
                [self.evaluator(f, j, obs) for f in self.spec.controllable_features]
                for j in prods
            ]
        )
        return fixed, control

    def prices(self, theta, obs):
        """
        Map coefficients and permitted features to bounded, cent-rounded fares (Eqs.
        18–19).
        """
        if len(theta) != len(self.spec.theta_bounds):
            raise ValueError("incorrect coefficient dimension")
        if not np.isfinite(theta).all():
            raise ValueError("pricing coefficients must be finite")
        result = []
        prods = self.c.products_of[self.i]
        markets = tuple(dict.fromkeys(self.c.spec.products[j].market for j in prods))
        size = len(self.spec.controllable_features)
        for local, j in enumerate(prods):
            p = self.c.spec.products[j]
            base = self.spec.base.get(p.id, p.reference_fare)
            base += self.spec.time_ramp * obs.t / (self.c.spec.horizon.n_periods - 1)
            base += sum(
                weight * self.evaluator(f, j, obs)
                for f, weight in self.spec.fixed_features
            )
            block = (
                0
                if self.spec.sharing == "carrier"
                else local
                if self.spec.sharing == "product"
                else markets.index(p.market)
            )
            adjustment = sum(
                theta[block * size + k] * self.evaluator(f, j, obs)
                for k, f in enumerate(self.spec.controllable_features)
            )
            result.append(
                round_fare(clip_fare(base + adjustment, p.fare_floor, p.fare_cap))
            )
        return np.array(result, dtype=np.float64)
