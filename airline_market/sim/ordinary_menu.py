"""A.10/A.5/B0/V9: prospective chronological ordinary choice inputs.

Analyst-only full net menu before each genuine ordinary choice. No likelihood,
prior inference, current private-mark inspection or additional utility forward.
"""

from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.ordinary_cohort import OrdinaryCohortMarketEngine
from airline_market.sim.passenger_accounts import _encode

RULE = "chronological-ordinary-menu-before-choice-v1"


@dataclass(frozen=True)
class OrdinaryMenuRow:
    person: object
    prices: tuple
    net_sellable: tuple
    native_columns: tuple
    native_class_weights: tuple


@dataclass(frozen=True)
class OrdinaryMenuPeriod:
    period: int
    rows: tuple


@dataclass(frozen=True)
class OrdinaryMenuSnapshot:
    native: object
    periods: tuple
    pending: object
    inflight: object
    spec_hash: str
    rule: str = RULE


class OrdinaryMenuMarketEngine(OrdinaryCohortMarketEngine):
    """One inherited dated chronological lifecycle plus prospective input rows.

    Random/mark-dependent admission is unsupported here. The hook does not read
    current taste_class, Gumbels or mixed marks before yielding the request.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("ordinary_priority") is not None:
            raise ValueError("chronological admission required for menu capture")
        self._menu_periods = []
        self._pending_menus = None
        self._menu_inflight = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._menu_periods = []
        self._pending_menus = None
        self._menu_inflight = None
        return super().reset(seed, tape=tape)

    def _track_requests(self, requests):
        for request in super()._track_requests(requests):
            period = self.world.t
            index = len(self.world.request_log)
            key = (period, request.gumbel_row, request.market)
            columns = (
                *self.compiled.market_products[request.market],
                len(self.compiled.spec.products),
            )
            weights = tuple(
                float(c.weight)
                for c in self.compiled.spec.segments[request.segment].classes
            )
            count = len(self._pending_menus)
            self._menu_inflight = (key, index, columns, weights)
            # Resume only after the one inherited native choice/log append.
            yield request
            if len(self.world.request_log) != index + 1:
                raise RuntimeError(
                    "one native ordinary choice per captured menu required"
                )
            if self._menu_inflight is not None or len(self._pending_menus) != count + 1:
                raise RuntimeError(
                    "one SAME-call native menu required per ordinary choice"
                )

    def _sellable_global(self):
        # The inherited call can synchronize held-seat ledgers. Delegate ONCE,
        # then copy its exact returned mask; never make an extra availability call.
        active = super()._sellable_global()
        if self._menu_inflight is not None:
            key, index, columns, weights = self._menu_inflight
            prices = tuple(float(p) for p in self._prices_global())
            mask = tuple(bool(v) for v in active)
            self._pending_menus.append((key, index, prices, mask, columns, weights))
            self._menu_inflight = None
        return active

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self._pending_menus is not None:
            raise ValueError("incomplete menu capture; no continuation or retry")
        if self.ordinary_priority is not None:
            raise ValueError("chronological admission required for menu capture")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        period = self.world.t
        self._pending_menus = []
        result = super().step(actions)
        ordinary = super().ordinary_periods()[-1]
        pending = tuple(self._pending_menus)
        if ordinary.period != period or len(ordinary.rows) != len(pending):
            raise RuntimeError("complete native ordinary cohort and menus differ")
        rows = []
        for person, (key, index, prices, mask, columns, weights) in zip(
            ordinary.rows, pending
        ):
            if person.key != key or person.native_log_index != index:
                raise RuntimeError("native ordinary identity and menu differ")
            rows.append(OrdinaryMenuRow(person, prices, mask, columns, weights))
        self._menu_periods.append(OrdinaryMenuPeriod(period, tuple(rows)))
        self._pending_menus = None
        self._menu_inflight = None
        return result

    def menu_periods(self):
        if self._pending_menus is not None:
            raise ValueError("incomplete menu capture; no partial query or salvage")
        return tuple(self._menu_periods)

    def snapshot(self):
        return OrdinaryMenuSnapshot(
            super().snapshot(),
            tuple(self._menu_periods),
            None if self._pending_menus is None else tuple(self._pending_menus),
            self._menu_inflight,
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not OrdinaryMenuSnapshot:
            raise TypeError("combined ordinary-menu snapshot required")
        if snapshot.rule != RULE or snapshot.spec_hash != self.compiled.spec_hash:
            raise ValueError("ordinary-menu capture contract differs")
        super().restore(snapshot.native)
        self._menu_inflight = snapshot.inflight
        self._menu_periods = list(snapshot.periods)
        self._pending_menus = (
            None if snapshot.pending is None else list(snapshot.pending)
        )


def ordinary_menu_record(engine):
    """Owned lossless NEW prospective record; no producer or likelihood replay."""
    if type(engine) is not OrdinaryMenuMarketEngine:
        raise TypeError("native chronological ordinary-menu producer required")
    periods = engine.menu_periods()
    return _encode(
        jsonable(
            deep_freeze(
                {
                    "plan_id": "A.10/A.5/B0/V9",
                    "capture_rule": RULE,
                    "spec_hash": engine.compiled.spec_hash,
                    "native_spec": jsonable(engine.compiled.spec),
                    "periods": [
                        {"period": p.period, "rows": [jsonable(r) for r in p.rows]}
                        for p in periods
                    ],
                    "chronological_native_menu_inputs_captured": True,
                    "native_population_class_prior_inferred": False,
                    "menu_independence_from_all_latent_history_authenticated": False,
                    "source_time_or_empirical_identity_authenticated": False,
                    "lifecycle_CS_or_delivered_service_identified": False,
                    "scientific_budget": None,
                }
            )
        )
    )
