"""A.10/A.5/B0/V9: SAME-call chronological ordinary menu information.

Prospective menu-only projection, not a complete passenger information set,
observability authentication, inferred belief or changed native choice policy.
"""

import hashlib
import json
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.sim.ordinary_menu import OrdinaryMenuMarketEngine
from airline_market.sim.passenger_accounts import _encode

RULE = "same-call-ordinary-market-menu-information-v1"
SCOPE = "current-market-menu-before-taste-utility-choice"


@dataclass(frozen=True)
class OrdinaryInformationOffer:
    product_id: str
    fare: float
    offered: bool


@dataclass(frozen=True)
class OrdinaryMenuInformation:
    market_id: str
    native_period: int
    compiled_opportunity_utc: str
    offers: tuple
    outside_available: bool = True
    scope: str = SCOPE


@dataclass(frozen=True)
class OrdinaryInformationRow:
    analyst_person: object
    menu_view: OrdinaryMenuInformation


@dataclass(frozen=True)
class OrdinaryInformationPeriod:
    period: int
    rows: tuple


@dataclass(frozen=True)
class OrdinaryInformationSnapshot:
    native: object
    periods: tuple
    pending: object
    spec_hash: str
    rule: str = RULE


class OrdinaryInformationMarketEngine(OrdinaryMenuMarketEngine):
    """Delegate ONE frozen native dated step; project its SAME-call menu.

    Before taste/utility/choice: captured key/current market fares/NET mask and
    compiled clock. Request generation may already assign a private class; this
    recorder never inspects it. Class/outcome join AFTER full closure.
    """

    def __init__(self, *args, **kwargs):
        self._information_periods = []
        self._pending_information = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._information_periods = []
        self._pending_information = None
        return super().reset(seed, tape=tape)

    def _make_information(self, native_menu):
        key, index, prices, mask, columns, _weights = native_menu
        market = self.compiled.spec.markets[key[2]].id
        offers = tuple(
            OrdinaryInformationOffer(
                self.compiled.spec.products[j].id, prices[j], mask[j]
            )
            for j in columns[:-1]
        )
        return (
            key,
            index,
            OrdinaryMenuInformation(
                market,
                key[0],
                self.demand_scenario.clock.at(key[0]).isoformat(),
                offers,
            ),
        )

    def _sellable_global(self):
        before = None if self._pending_menus is None else len(self._pending_menus)
        active = super()._sellable_global()
        if before is not None and len(self._pending_menus) != before:
            if len(self._pending_menus) != before + 1:
                raise RuntimeError("one SAME-call ordinary menu required")
            self._pending_information.append(
                self._make_information(self._pending_menus[-1])
            )
        return active

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self._pending_information is not None:
            raise ValueError("incomplete information capture; no continuation or retry")
        if self.ordinary_priority is not None:
            raise ValueError("chronological admission required for information capture")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        for name in ("booking_clock", "shock_clock"):
            clock = getattr(self, name, None)
            if clock is not None and clock != self.demand_scenario.clock:
                raise ValueError("native clock and compiled information clock differ")
        period = self.world.t
        self._pending_information = []
        result = super().step(actions)
        menus = super().menu_periods()[-1]
        if menus.period != period or len(menus.rows) != len(self._pending_information):
            raise RuntimeError("complete ordinary menus and information differ")
        rows = []
        for source, (key, index, view) in zip(menus.rows, self._pending_information):
            if source.person.key != key or source.person.native_log_index != index:
                raise RuntimeError(
                    "native original-person information identity differs"
                )
            rows.append(OrdinaryInformationRow(source.person, view))
        self._information_periods.append(OrdinaryInformationPeriod(period, tuple(rows)))
        self._pending_information = None
        return result

    def information_periods(self):
        if self._pending_information is not None:
            raise ValueError(
                "incomplete information capture; no partial query or salvage"
            )
        return tuple(self._information_periods)

    def snapshot(self):
        return OrdinaryInformationSnapshot(
            super().snapshot(),
            tuple(self._information_periods),
            None
            if self._pending_information is None
            else tuple(self._pending_information),
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not OrdinaryInformationSnapshot:
            raise TypeError("combined ordinary-information snapshot required")
        if snapshot.rule != RULE or snapshot.spec_hash != self.compiled.spec_hash:
            raise ValueError("ordinary-information capture contract differs")
        super().restore(snapshot.native)
        self._information_periods = list(snapshot.periods)
        self._pending_information = (
            None if snapshot.pending is None else list(snapshot.pending)
        )


def ordinary_information_view(row):
    """Owned menu-only view; no source replay/private identity/class/outcome."""
    if type(row) is not OrdinaryInformationRow:
        raise TypeError("ordinary-information row required")
    return _encode(jsonable(row.menu_view))


def ordinary_information_key(row):
    """Hash ONLY this narrow view, not complete knowledge/source/CRN identity."""
    payload = json.dumps(
        ordinary_information_view(row),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def ordinary_information_record(engine):
    """NEW complete receipt export; analyst context separate from menu views."""
    if type(engine) is not OrdinaryInformationMarketEngine:
        raise TypeError("native ordinary-information producer required")
    periods = engine.information_periods()
    return {
        "plan_id": "A.10/A.5/B0/V9",
        "capture_rule": RULE,
        "analyst_context": _encode(
            jsonable(
                {
                    "spec_hash": engine.compiled.spec_hash,
                    "native_spec": engine.compiled.spec,
                    "identity_by_period": tuple(
                        {
                            "period": p.period,
                            "people": tuple(r.analyst_person for r in p.rows),
                        }
                        for p in periods
                    ),
                }
            )
        ),
        "menu_views_by_period": tuple(
            {
                "period": p.period,
                "views": tuple(ordinary_information_view(r) for r in p.rows),
                "view_keys": tuple(ordinary_information_key(r) for r in p.rows),
            }
            for p in periods
        ),
        "same_call_native_menu_projection_captured": True,
        "complete_passenger_information_set_or_public_visibility_authenticated": False,
        "native_actor_observations_or_choice_changed": False,
        "subjective_or_true_future_law_inferred": False,
        "lifecycle_utility_CS_or_delivered_service_identified": False,
        "scientific_budget": None,
    }
