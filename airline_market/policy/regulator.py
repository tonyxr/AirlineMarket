"""A.8/B0/V9: opt-in reference-fare caps on native price posting.

This is the execution constraint and reference-window component. Native
request order, choice, booking, cancellation and accounting are inherited.
Rationing and instrument-aware planner optimization are separate components.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_FLOOR, Decimal
from statistics import median

import numpy as np

from airline_market.sim.engine import MarketEngine


def _utc(value):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("reference timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReferenceQuote:
    """Observed gross quote keyed by exact origin, destination and product class."""

    key: tuple[str, str, str]
    observed_at: datetime
    fare: object


def reference_medians(quotes, reveal):
    """28-day pre-reveal observed-quote medians; never fill missing days with zero.

    Values at/after the reveal are excluded before their fare is inspected.
    Quality counts and observed-day coverage are retained for every in-window
    key. This does not establish complete exposure, bookability or inventory.
    """
    stop = _utc(reveal)
    start = stop - timedelta(days=28)
    groups = {}
    excluded = 0
    for quote in quotes:
        when = _utc(quote.observed_at)
        if not start <= when < stop:
            excluded += 1
            continue
        if len(quote.key) != 3 or any(
            not isinstance(k, str) or not k for k in quote.key
        ):
            raise ValueError(
                "reference key requires origin, destination and product class"
            )
        group = groups.setdefault(
            quote.key,
            {
                "values": [],
                "days": set(),
                "missing": 0,
                "nonfinite": 0,
                "nonpositive": 0,
            },
        )
        group["days"].add(when.date())
        if quote.fare is None:
            group["missing"] += 1
            continue
        fare = Decimal(str(quote.fare))
        if not fare.is_finite():
            group["nonfinite"] += 1
        elif fare <= 0:
            group["nonpositive"] += 1
        else:
            group["values"].append(fare)
    rows = []
    for key, group in sorted(groups.items()):
        value = median(group["values"]) if group["values"] else None
        rows.append(
            {
                "key": list(key),
                "median_fare_decimal": None if value is None else str(value),
                "usable_quotes": len(group["values"]),
                "missing_quotes": group["missing"],
                "nonfinite_quotes": group["nonfinite"],
                "nonpositive_quotes": group["nonpositive"],
                "observed_days": len(group["days"]),
            }
        )
    return {
        "plan_id": "A.8/B0/V9",
        "start_inclusive": start.isoformat(),
        "stop_exclusive": stop.isoformat(),
        "groups": rows,
        "out_of_window_quotes": excluded,
        "exposure_scope": "Observed quotes only; missing days are not zero demand or nonpurchase",
    }


def _interval(start, stop):
    if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
        raise ValueError("cap intervals require integer 0 <= start < stop")


@dataclass(frozen=True)
class CapWindow:
    """Reference-relative constraint on [start, stop), in native booking periods."""

    start: int
    stop: int
    reference_fares: tuple[tuple[str, object], ...]
    multiplier: object = "1.30"
    requires_emergency: bool = False

    def __post_init__(self):
        _interval(self.start, self.stop)
        multiplier = Decimal(str(self.multiplier))
        refs = tuple(
            sorted((key, Decimal(str(value))) for key, value in self.reference_fares)
        )
        if not multiplier.is_finite() or multiplier <= 0:
            raise ValueError("cap multiplier must be finite and positive")
        if len({key for key, _ in refs}) != len(refs) or any(
            not isinstance(key, str) or not key or not value.is_finite() or value <= 0
            for key, value in refs
        ):
            raise ValueError(
                "cap references must have unique products and positive finite fares"
            )
        if type(self.requires_emergency) is not bool:
            raise ValueError("emergency requirement must be a public boolean")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "reference_fares", refs)

    def cent_limits(self):
        """Floor to a cent so the executed quote never exceeds the declared cap."""
        return {
            key: float(
                (value * self.multiplier).quantize(
                    Decimal("0.01"), rounding=ROUND_FLOOR
                )
            )
            for key, value in self.reference_fares
        }


@dataclass(frozen=True)
class CapPolicy:
    """Immutable public windows; overlapping constraints use their lowest cap."""

    windows: tuple[CapWindow, ...] = ()
    declared_emergencies: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        windows = tuple(self.windows)
        declarations = tuple(tuple(interval) for interval in self.declared_emergencies)
        if any(not isinstance(window, CapWindow) for window in windows):
            raise ValueError("policy windows must be CapWindow records")
        for start, stop in declarations:
            _interval(start, stop)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "declared_emergencies", declarations)

    def limits(self, period):
        """Read the public clock/declarations only, never a hidden world state."""
        emergency = any(
            start <= period < stop for start, stop in self.declared_emergencies
        )
        limits = {}
        for window in self.windows:
            if window.start <= period < window.stop and (
                emergency or not window.requires_emergency
            ):
                for key, value in window.cent_limits().items():
                    limits[key] = min(limits.get(key, value), value)
        return limits


NO_CAP = CapPolicy()


class RegulatedMarketEngine(MarketEngine):
    """Opt-in cap enforcement after each native posting and before native choice.

    An active cap takes precedence over the ordinary pricing floor. With no
    binding cap, the native price array and all world fields are untouched.
    Both leader and follower posting paths call this same native hook.
    """

    def __init__(
        self, compiled, controllers=None, *, cap_policy=NO_CAP, check_invariants=True
    ):
        super().__init__(compiled, controllers, check_invariants=check_invariants)
        if not isinstance(cap_policy, CapPolicy):
            raise TypeError("explicit CapPolicy required")
        targets = {
            key for window in cap_policy.windows for key, _ in window.reference_fares
        }
        if targets - set(compiled.product_index):
            raise ValueError("cap targets an unknown product")
        self.cap_policy = cap_policy

    def _action_prices(self, i, action, obs):
        super()._action_prices(i, action, obs)
        limits = self.cap_policy.limits(obs.t)
        if not limits:
            return
        state = self.world.carriers[i]
        products = self.compiled.products_of[i]
        capped = state.posted_prices.copy()
        for local, j in enumerate(products):
            key = self.compiled.spec.products[j].id
            if key in limits:
                capped[local] = min(capped[local], limits[key])
        if not np.array_equal(capped, state.posted_prices):
            state.posted_prices = capped

    def branch(self):
        """Retain the immutable intervention when cloning native mutable state."""
        other = RegulatedMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
