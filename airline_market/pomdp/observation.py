"""Carrier-local observations and stable vector schemas (Eqs. 16, 24)."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping
import numpy as np
from ..core.errors import InformationLeakError
from ..core.hashing import schema_hash as hash_schema
from ..core.snapshot import FrozenMap
from ..sim.state import Booking
from ..sim.ledger import inventory
from ..sim.exogenous import public_signal
from ..sim.quotes import shop


@dataclass(frozen=True)
class QuoteObs:
    """
    Permitted displayed fare, availability and seat information for one product (Eqs.
    24–27).
    """

    product: int
    period_posted: int
    fare: float | None
    available: bool | None
    seats: int | None
    missing: bool


@dataclass(frozen=True)
class PrivilegedState:
    """
    Explicit oracle/critic diagnostics, never part of a public actor arm (Eqs. 24–27).
    """

    capacities: tuple
    bookings: tuple
    controller_memory: tuple
    exo: int


@dataclass(frozen=True)
class Observation:
    """
    Immutable permitted carrier history with no seed, tape or world reference (Eqs.
    24–27).
    """

    t: int
    carrier: int
    arm: str
    own_capacity: tuple[int, ...]
    own_bookings: tuple[Booking, ...]
    own_cash: float
    own_refunds: float
    own_previous_prices: tuple[float, ...]
    own_posted_prices: tuple[float, ...] | None
    public_exo: Mapping[str, float]
    quotes: tuple[QuoteObs, ...]
    own_inventory: tuple[int, ...]
    privileged: PrivilegedState | None = None
    own_purchases: tuple[tuple[int, float], ...] = ()

    def require_privileged(self) -> PrivilegedState:
        """
        Expose hidden diagnostics only for an explicitly privileged arm (Eqs. 24–27).
        """
        if self.privileged is None:
            raise InformationLeakError(
                "hidden state requested by a public-information actor"
            )
        return self.privileged

    def __getattr__(self, name):
        """
        Forward supported interface queries or reject forbidden hidden-state access
        (Eqs. 24–27).
        """
        if name in ("seed", "tape", "world", "rival_inventory", "rival_bookings"):
            raise InformationLeakError(f"{name} is not an actor observation")
        raise AttributeError(name)


class ObservationBuilder:
    """
    Construct the declared scoped view while redacting unobserved booking marks (Eqs.
    24–27).
    """

    def __init__(self, compiled, arm, request_specs=None):
        """
        Bind declared inputs and initialize ObservationBuilder bookkeeping (Eqs. 24–27).
        """
        self.compiled = compiled
        self.arm = arm
        self.requests = {
            q.id: (k, q) for k, q in enumerate(compiled.spec.quotes.requests)
        }

    def build(self, world, history, tape, t: int | None = None) -> Observation:
        """
        Build own ledger/public signals and only the arm's permitted quotes (Eqs.
        24–27).
        """
        c = self.compiled
        i = c.carrier_index[self.arm.observer]
        state = world.carriers[i]
        t = world.t if t is None else t
        quotes = []
        for qid in self.arm.requests:
            index, request = self.requests[qid]
            quotes.extend(shop(history, request, t, tape, c, index))
        # Internal global IDs and latent taste marks are not customer-facing data.
        bookings = tuple(
            replace(b, record_id=k, arrival_rank=-1, segment=-1, taste_class=-1)
            for k, b in enumerate(state.bookings)
        )
        privileged = None
        if self.arm.privileged:
            privileged = PrivilegedState(
                tuple(tuple(int(x) for x in cs.capacity) for cs in world.carriers),
                tuple(tuple(cs.bookings) for cs in world.carriers),
                tuple(cs.controller_memory for cs in world.carriers),
                world.exo,
            )
        public = dict(public_signal(c, world.exo))
        joint_public = {
            leg
            for prior in c.spec.capacity_priors
            if prior.public
            for leg in prior.legs
        }
        for l, leg in enumerate(c.spec.legs):
            if leg.capacity.public or leg.id in joint_public:
                owner = int(c.leg_carrier[l])
                local = c.legs_of[owner].index(l)
                public["capacity/" + leg.id] = float(
                    world.carriers[owner].capacity[local]
                )
        return Observation(
            t,
            i,
            self.arm.name,
            tuple(map(int, state.capacity)),
            bookings,
            state.cash,
            state.refunds,
            tuple(map(float, state.previous_prices)),
            tuple(map(float, state.posted_prices)),
            FrozenMap(public),
            tuple(quotes),
            tuple(map(int, inventory(state, c, i))),
            privileged,
            tuple((b.product, b.fare) for b in world.last_purchases if b.carrier == i),
        )


class Vectorizer:
    """
    Stable ordered numerical observation contract with a schema fingerprint (Eqs.
    24–27).
    """

    def __init__(self, compiled, arm, feature_spec=None):
        """Bind declared inputs and initialize Vectorizer bookkeeping (Eqs. 24–27)."""
        self.compiled = compiled
        self.arm = arm
        self.i = compiled.carrier_index[arm.observer]
        requests = {q.id: q for q in compiled.spec.quotes.requests}
        self.targets = tuple(
            dict.fromkeys(p for q in arm.requests for p in requests[q].target_products)
        )
        self._names = tuple(
            ["clock"]
            + [
                f"inventory/{compiled.spec.legs[l].id}"
                for l in compiled.legs_of[self.i]
            ]
            + [
                f"bookings/{compiled.spec.products[j].id}"
                for j in compiled.products_of[self.i]
            ]
            + [
                f"quote/{p}/{key}"
                for p in self.targets
                for key in ("fare", "available", "missing", "seats")
            ]
            + [f"exo/{e.name}" for e in compiled.spec.exogenous.states if e.public]
        )

    def names(self):
        """Return stable feature names in the encoded observation order (Eqs. 24–27)."""
        return self._names

    def schema_hash(self):
        """Fingerprint the exact observation feature contract (Eqs. 24–27)."""
        return hash_schema(self._names)

    def __call__(self, obs: Observation) -> np.ndarray:
        """
        Encode permitted observation fields in the fingerprinted feature order (Eqs.
        24–27).
        """
        c = self.compiled
        values = [obs.t / c.spec.horizon.n_periods]
        values += [x / max(1, k) for x, k in zip(obs.own_inventory, obs.own_capacity)]
        values += [
            sum(b.product == j for b in obs.own_bookings) for j in c.products_of[self.i]
        ]
        quotes = {q.product: q for q in obs.quotes}
        names = {
            **c.product_index,
            **{b.id: len(c.spec.products) + k for k, b in enumerate(c.spec.background)},
        }
        for p in self.targets:
            j = names[p]
            q = quotes.get(j)
            ref = (
                c.spec.products[j].reference_fare if j < len(c.spec.products) else 100.0
            )
            values += [
                q.fare / ref if q and q.fare is not None else 0.0,
                float(bool(q.available)) if q else 0.0,
                float(q is None or q.missing),
                q.seats / 9 if q and q.seats is not None else 0.0,
            ]
        values += [
            obs.public_exo.get("state:" + e.name, 0.0)
            for e in c.spec.exogenous.states
            if e.public
        ]
        return np.asarray(values, dtype=np.float32)
