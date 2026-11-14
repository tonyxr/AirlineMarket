"""Opt-in, price-independent excluded-ticket reservations on core resources.

Technical single-leg products carry physical bookings only. They have no
customer arrivals, competitive journeys or attributed revenue. Joint groups
are admitted atomically and canceled as a whole after a capacity disruption.
The resulting objective is strategic-market contribution conditional on this
external load, not total airline profit. Existing samplers/solvers are unchanged.
"""

import copy
import json
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..spec import dump_spec, load_spec
from .engine import MarketEngine
from .ledger import book, inventory

PREFIX = "__CORE_LOAD__"


@dataclass(frozen=True)
class LoadEvent:
    """One external ticket's declared dated core-resource footprint before pricing."""

    id: int
    period: int
    legs: tuple[str, ...]
    inclusion_uniform: float = 0.0


@dataclass(frozen=True)
class LoadSnapshot:
    """Retain the physical tape and all external-reservation state for branching."""

    native: object
    applied_periods: frozenset
    groups: tuple
    canceled_groups: frozenset
    counters: tuple


def augment_load_products(spec, carriers):
    """Add zero-demand ledger products while preserving original product indices.

    Their positive placeholder posted fare satisfies the unchanged spec schema;
    no customer can arrive in these markets and actual load receipts are zero.
    Background components have lower disruption priority than strategic products.
    """
    if spec.demand.mode != "poisson":
        raise ValueError("shared background augmentation requires Poisson demand")
    if spec.resolution.penalty_per_cancellation != 0:
        raise ValueError("background cancellation penalties are not identified; require zero")
    if any(p.sharing != "carrier" for p in spec.pricing):
        raise ValueError("load augmentation currently preserves carrier-shared coefficient classes")
    data = json.loads(dump_spec(spec))
    if any(p["id"].startswith(PREFIX) for p in data["products"]):
        raise ValueError("background ledger products already present")
    priority = max(p.priority for p in spec.products) + 1
    for leg in spec.legs:
        if leg.carrier not in carriers:
            continue
        identifier = PREFIX + leg.id
        data["markets"].append({"id": identifier, "origin": leg.origin, "dest": leg.dest,
                                "in_segment": False, "label": "technical load ledger only"})
        data["products"].append({"id": identifier, "carrier": leg.carrier,
            "market": identifier, "legs": [leg.id], "service_cost": 0.0,
            "fare_floor": 1.0, "fare_cap": 1.0, "reference_fare": 1.0,
            "priority": priority, "controlled": False})
        data["demand"]["intensity"][identifier] = {"constant": 0.0}
        priority += 1
    data["metadata"]["shared_core_load"] = {
        "technical_prefix": PREFIX, "carriers": list(carriers),
        "objective_scope": "strategic-market gross contribution conditional on external load",
        "background_revenue_identified": False,
        "disruption_priority": "strategic products first, background groups after; assumption",
        "solver_scope": "native physical sampler only; not integrated into exact/PF/MPC or training"}
    return load_spec(data)


class SharedLoadEngine(MarketEngine):
    """Execute immutable external load events before each period's observations."""

    def __init__(self, compiled, events=(), controllers=None, *, fraction=1.0,
                 check_invariants=True):
        """Bind the declared load tape without exposing future events to actors."""
        super().__init__(compiled, controllers, check_invariants=check_invariants)
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("load fraction must lie in [0,1]")
        if compiled.spec.resolution.penalty_per_cancellation != 0:
            raise ValueError("shared background requires zero cancellation penalties")
        self.events = tuple(events)
        self.fraction = float(fraction)
        self.load_products = {p.legs[0]: j for j, p in enumerate(compiled.spec.products)
                              if p.id.startswith(PREFIX)}
        ids = set()
        for event in self.events:
            if (event.id in ids or not 0 <= event.period < compiled.spec.horizon.n_periods
                    or not event.legs or any(l not in self.load_products for l in event.legs)
                    or not 0 <= event.inclusion_uniform < 1):
                raise ValueError("invalid or duplicate external load event")
            ids.add(event.id)
        self.by_period = {}
        for event in self.events:
            self.by_period.setdefault(event.period, []).append(event)
        self.applied_periods, self.groups, self.canceled_groups = set(), {}, set()
        self.load_counts = Counter()

    def reset(self, seed=None, *, tape=None):
        """Clear external episode state while retaining the immutable event schedule."""
        self.applied_periods, self.groups, self.canceled_groups = set(), {}, set()
        self.load_counts = Counter()
        super().reset(seed, tape=tape)

    def _apply_load(self):
        """Admit all or none of every footprint, including cross-network components."""
        t, c = self.world.t, self.compiled
        if t in self.applied_periods or self.terminal():
            return
        self.applied_periods.add(t)
        for event in self.by_period.get(t, ()):
            if event.inclusion_uniform >= self.fraction:
                self.load_counts["fraction_excluded"] += 1
                continue
            self.load_counts["attempted_groups"] += 1
            need = Counter(event.legs)
            stocks = {i: inventory(s, c, i) for i, s in enumerate(self.world.carriers)}
            if any(stocks[int(c.leg_carrier[c.leg_index[l]])][
                    c.legs_of[int(c.leg_carrier[c.leg_index[l]])].index(c.leg_index[l])] < n
                    for l, n in need.items()):
                self.load_counts["capacity_rejected_groups"] += 1
                continue
            group = []
            request = SimpleNamespace(rank=-1, segment=-1, taste_class=-1)
            for leg in event.legs:
                j = self.load_products[leg]
                i = int(c.product_carrier[j])
                b = book(self.world, i, j, 0.0, request, c)
                group.append((i, b.record_id))
                self.load_counts["accepted_components"] += 1
            self.groups[event.id] = tuple(group)
            self.load_counts["accepted_groups"] += 1
            for leg, amount in need.items():
                global_leg = c.leg_index[leg]
                carrier = int(c.leg_carrier[global_leg])
                local = c.legs_of[carrier].index(global_leg)
                if stocks[carrier][local] == amount:
                    self.world.sellout_period.setdefault(global_leg, t)

    def begin_period(self, actions=None):
        """Apply only currently due events, before quotes or coefficient decisions."""
        if self.world is not None:
            self._apply_load()
        return super().begin_period(actions)

    def reconcile_load_groups(self):
        """Release every surviving component if native resolution cancels any member."""
        live = {(i, b.record_id) for i, s in enumerate(self.world.carriers) for b in s.bookings}
        for identifier, group in self.groups.items():
            if identifier in self.canceled_groups or all(key in live for key in group):
                continue
            self.canceled_groups.add(identifier)
            for i, record in group:
                if (i, record) not in live:
                    continue
                state = self.world.carriers[i]
                state.bookings = [b for b in state.bookings if b.record_id != record]
                state.canceled += 1
                live.remove((i, record))
        self.load_counts["canceled_groups"] = len(self.canceled_groups)

    def step(self, actions=None, *, _requests=None, _choices=None):
        """Retain native strategic shopping and settle coupled load after disruptions."""
        result = super().step(actions, _requests=_requests, _choices=_choices)
        self.reconcile_load_groups()
        return result

    def metrics(self):
        """Separate technical component counts from strategic contribution and share."""
        result = super().metrics()
        c = self.compiled
        canceled = Counter(i for g in self.canceled_groups for i, _ in self.groups[g])
        loads = {}
        for i, carrier in enumerate(c.spec.carriers):
            active = [b for b in self.world.carriers[i].bookings
                      if c.spec.products[b.product].id.startswith(PREFIX)]
            value = result["carriers"][carrier.code]
            excluded_cost = sum(float(c.product_cost[b.product]) for b in active)
            objective = next(o for o in c.spec.objectives if o.carrier == carrier.code)
            value["service_cost"] -= excluded_cost
            value["profit"] += excluded_cost
            value["objective"] += excluded_cost / objective.profit_scale
            value["canceled"] -= canceled[i]
            value["cancellations"] -= canceled[i]
            loads[carrier.code] = {"active_components": len(active),
                "canceled_components": canceled[i], "attributed_revenue": None,
                "attributed_cost": None}
        focal = c.spec.metadata.get("focal", c.spec.carriers[0].code)
        for key in ("profit", "objective", "cancellations"):
            result[key] = result["carriers"][focal][key]
        result["rival_profit"] = sum(v["profit"] for k, v in result["carriers"].items() if k != focal)
        result["shared_core_load"] = {**dict(self.load_counts), "by_carrier": loads,
            "objective_scope": "strategic-market gross contribution conditional on external load; no total airline or net-profit claim"}
        return result

    def snapshot(self):
        """Save load admission/coupling state together with native physical state."""
        return LoadSnapshot(super().snapshot(), frozenset(self.applied_periods),
                            tuple(self.groups.items()), frozenset(self.canceled_groups),
                            tuple(self.load_counts.items()))

    def restore(self, snapshot):
        """Restore without redrawing or readmitting external reservations."""
        super().restore(snapshot.native)
        self.applied_periods = set(snapshot.applied_periods)
        self.groups = dict(snapshot.groups)
        self.canceled_groups = set(snapshot.canceled_groups)
        self.load_counts = Counter(dict(snapshot.counters))

    def branch(self):
        """Clone private execution state while sharing the same immutable load tape."""
        other = SharedLoadEngine(self.compiled, self.events,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            fraction=self.fraction, check_invariants=self.check_invariants)
        other.restore(self.snapshot())
        return other
