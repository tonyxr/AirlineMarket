"""A.14/B0/V9: public-input designed-trigger rule and native cap execution.

Mechanical fares are an explicit frozen public-input table. This module does
not generate or validate the B2 mechanical counterfactual or tune B4 thresholds.
All windows use native periods, with no inferred day/hour mapping.
"""

import copy
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

import numpy as np

from airline_market.policy.reserve import ReserveMarketEngine


def _fares(rows, *, allow_zero=False):
    result = tuple(sorted((key, Decimal(str(value))) for key, value in rows))
    if (
        not result
        or len({k for k, _ in result}) != len(result)
        or any(
            not isinstance(k, str)
            or not k
            or not v.is_finite()
            or (v < 0 if allow_zero else v <= 0)
            for k, v in result
        )
    ):
        raise ValueError("unique product IDs and positive finite public fares required")
    return result


def _stock(rows):
    result = tuple(sorted(tuple(row) for row in rows))
    if (
        not result
        or len({k for k, _ in result}) != len(result)
        or any(
            not isinstance(k, str) or not k or type(n) is not int or n < 0
            for k, n in result
        )
    ):
        raise ValueError("unique public leg IDs and nonnegative integer stock required")
    return result


@dataclass(frozen=True, slots=True)
class PublicTriggerReport:
    period: int
    baseline_capacity: tuple
    observed_capacity: tuple
    remaining_seats: tuple
    reference_fares: tuple
    declared_emergency: bool = False

    def __post_init__(self):
        if (
            type(self.period) is not int
            or self.period < 0
            or type(self.declared_emergency) is not bool
        ):
            raise ValueError("public clock and explicit emergency boolean required")
        for name in ["baseline_capacity", "observed_capacity", "remaining_seats"]:
            object.__setattr__(self, name, _stock(getattr(self, name)))
        base, observed, remaining = (
            dict(getattr(self, name))
            for name in ["baseline_capacity", "observed_capacity", "remaining_seats"]
        )
        if (
            set(base) != set(observed)
            or set(base) != set(remaining)
            or any(remaining[k] > observed[k] for k in base)
        ):
            raise ValueError(
                "public capacity reports must cover the same legs; remaining <= observed capacity"
            )
        object.__setattr__(self, "reference_fares", _fares(self.reference_fares))

    @property
    def loss_fraction(self):
        # Installed-capacity loss, not booking-induced inventory depletion.
        base, current = dict(self.baseline_capacity), dict(self.observed_capacity)
        total = sum(base.values())
        return (
            Decimal(sum(max(0, n - current[k]) for k, n in base.items()))
            / Decimal(total)
            if total
            else Decimal(0)
        )

    @property
    def benchmark_key(self):
        return self.period, self.remaining_seats, self.reference_fares


@dataclass(frozen=True, slots=True)
class MechanicalFareRow:
    period: int
    remaining_seats: tuple
    reference_fares: tuple
    mechanical_fares: tuple

    def __post_init__(self):
        if type(self.period) is not int or self.period < 0:
            raise ValueError("explicit native period required")
        object.__setattr__(self, "remaining_seats", _stock(self.remaining_seats))
        object.__setattr__(self, "reference_fares", _fares(self.reference_fares))
        object.__setattr__(self, "mechanical_fares", _fares(self.mechanical_fares))

    @property
    def key(self):
        return self.period, self.remaining_seats, self.reference_fares


@dataclass(frozen=True, slots=True)
class PublicMechanicalTable:
    """Deterministic lookup keyed ONLY by the declared public mechanical inputs.

    Source hashes identify the frozen pre-shock demand/rival model and the
    supplied counterfactual table. They do not certify its economic validity.
    Unrepresented public states raise; no nearest-state or true-world fallback.
    """

    rows: tuple[MechanicalFareRow, ...]
    pre_shock_demand_sha256: str
    frozen_rival_policy_sha256: str

    def __post_init__(self):
        rows = tuple(self.rows)
        if not rows or any(type(r) is not MechanicalFareRow for r in rows):
            raise ValueError("explicit public mechanical table rows required")
        if len({r.key for r in rows}) != len(rows):
            raise ValueError("duplicate public benchmark key")
        targets = {k for k, _ in rows[0].mechanical_fares}
        if any({k for k, _ in r.mechanical_fares} != targets for r in rows):
            raise ValueError("mechanical table target products must remain fixed")
        stock_keys = {k for k, _ in rows[0].remaining_seats}
        reference_keys = {k for k, _ in rows[0].reference_fares}
        if any(
            {k for k, _ in r.remaining_seats} != stock_keys
            or {k for k, _ in r.reference_fares} != reference_keys
            for r in rows
        ):
            raise ValueError("public benchmark stock/reference scope must remain fixed")
        for value in [self.pre_shock_demand_sha256, self.frozen_rival_policy_sha256]:
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(x not in "0123456789abcdef" for x in value)
            ):
                raise ValueError(
                    "explicit SHA256 provenance for pre-shock demand/rival required"
                )
        object.__setattr__(self, "rows", rows)

    def fares(self, report):
        if type(report) is not PublicTriggerReport:
            raise TypeError("public trigger report required; no world or hidden state")
        for row in self.rows:
            if row.key == report.benchmark_key:
                return dict(row.mechanical_fares)
        raise KeyError("public state absent from frozen mechanical table")


@dataclass(frozen=True, slots=True)
class TriggerRule:
    start: int
    stop: int
    kappa: Decimal
    tau: Decimal
    window_periods: int

    def __post_init__(self):
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop
        ):
            raise ValueError("explicit ordered native window required")
        if type(self.window_periods) is not int or self.window_periods < 1:
            raise ValueError("positive integer activation window required")
        k, t = Decimal(str(self.kappa)), Decimal(str(self.tau))
        if not k.is_finite() or not 0 <= k <= 1 or t.is_nan() or t < 0:
            raise ValueError(
                "0 <= kappa <= 1 and nonnegative tau (including infinity) required"
            )
        object.__setattr__(self, "kappa", k)
        object.__setattr__(self, "tau", t)


class DesignedTrigger:
    """Fixed activation windows; no extension while active, re-arm after expiry.

    Each product is triggered by its native posted public fare exceeding the
    unrounded mechanical threshold, together with capacity loss >= kappa OR a
    declaration. Active products use the current public benchmark until expiry.
    Gross removed seats / baseline seats in the declared leg scope defines loss.
    """

    def __init__(self, rule, benchmark):
        if (
            type(rule) is not TriggerRule
            or type(benchmark) is not PublicMechanicalTable
        ):
            raise TypeError("explicit rule and public mechanical table required")
        self.rule, self.benchmark = rule, benchmark
        self.active_until = {}
        self.history = []
        self.last_period = None

    def evaluate(self, report, posted_fares):
        if type(report) is not PublicTriggerReport:
            raise TypeError("public trigger report required")
        period = report.period
        if self.last_period is not None and period < self.last_period:
            raise ValueError("public trigger clock cannot go backwards without restore")
        self.last_period = period
        rule = self.rule
        if not rule.start <= period < rule.stop or rule.tau.is_infinite():
            return {}
        mechanical = self.benchmark.fares(report)
        candidates = _fares(posted_fares, allow_zero=True)
        signal = report.loss_fraction >= rule.kappa or report.declared_emergency
        caps = {}
        for product, posted in candidates:
            if product not in mechanical:
                continue
            threshold = mechanical[product] * (1 + rule.tau)
            active = period < self.active_until.get(product, -1)
            fired = bool(not active and signal and posted > threshold)
            if fired:
                self.active_until[product] = min(
                    rule.stop, period + rule.window_periods
                )
            active = active or fired
            if active:
                caps[product] = float(
                    threshold.quantize(Decimal(".01"), rounding=ROUND_FLOOR)
                )
            self.history.append(
                {
                    "period": period,
                    "product": product,
                    "posted_fare": str(posted),
                    "mechanical_fare": str(mechanical[product]),
                    "raw_threshold": str(threshold),
                    "public_loss_fraction": str(report.loss_fraction),
                    "declared_emergency": report.declared_emergency,
                    "capacity_or_declaration": bool(signal),
                    "fired_now": fired,
                    "active": active,
                    "active_until": self.active_until.get(product),
                    "cap": caps.get(product),
                }
            )
        return caps


@dataclass(frozen=True)
class TriggerSnapshot:
    native: object
    active_until: tuple
    history: tuple
    reports: tuple
    last_period: int | None


class TriggerMarketEngine(ReserveMarketEngine):
    """Opt-in physical cap execution; old PF/exact/MPC kernels are not adopted.

    A regulator supplies public reports explicitly. The engine never fills them
    from private world inventory. Existing cap/exit/rebooking/rescue/reserve
    executors remain composable; actors still receive only native observations.
    """

    def __init__(
        self,
        compiled,
        controllers=None,
        *,
        trigger_rule=None,
        mechanical_benchmark=None,
        **kwargs,
    ):
        super().__init__(compiled, controllers, **kwargs)
        if (trigger_rule is None) != (mechanical_benchmark is None):
            raise ValueError("both explicit trigger rule and mechanical table required")
        self.trigger = (
            None
            if trigger_rule is None
            else DesignedTrigger(trigger_rule, mechanical_benchmark)
        )
        if self.trigger:
            targets = {
                k for r in mechanical_benchmark.rows for k, _ in r.mechanical_fares
            }
            if targets - set(compiled.product_index):
                raise ValueError("unknown trigger target")
            if any(
                {k for k, _ in row.remaining_seats} - set(compiled.leg_index)
                or {k for k, _ in row.reference_fares} - set(compiled.product_index)
                for row in mechanical_benchmark.rows
            ):
                raise ValueError("unknown public benchmark leg or quote product")
            if trigger_rule.stop > compiled.spec.horizon.n_periods:
                raise ValueError("trigger window outside native horizon")
        self._regulator_reports = {}

    def reset(self, seed=None, *, tape=None):
        self._regulator_reports = {}
        if self.trigger:
            self.trigger = DesignedTrigger(self.trigger.rule, self.trigger.benchmark)
        return super().reset(seed, tape=tape)

    def publish_regulator_report(self, report):
        if type(report) is not PublicTriggerReport:
            raise TypeError("only explicit public report accepted")
        if self.world is None or report.period != self.world.t:
            raise ValueError("regulator report must match current native period")
        old = self._regulator_reports.get(report.period)
        if old is not None and old != report:
            raise ValueError("published report is immutable within its period")
        self._regulator_reports[report.period] = report

    def _action_prices(self, i, action, obs):
        trigger = self.trigger
        enabled = bool(
            trigger
            and trigger.rule.start <= obs.t < trigger.rule.stop
            and not trigger.rule.tau.is_infinite()
        )
        report = self._regulator_reports.get(obs.t)
        if enabled and report is None:
            raise ValueError("fresh public regulator report required")
        # Check lookup before mutating native prices; no private-state fallback.
        if enabled:
            trigger.benchmark.fares(report)
        super()._action_prices(i, action, obs)
        if not enabled:
            return
        c = self.compiled
        state = self.world.carriers[i]
        candidates = tuple(
            (c.spec.products[j].id, float(state.posted_prices[k]))
            for k, j in enumerate(c.products_of[i])
        )
        caps = trigger.evaluate(report, candidates)
        if caps:
            updated = state.posted_prices.copy()
            for k, j in enumerate(c.products_of[i]):
                key = c.spec.products[j].id
                if key in caps:
                    updated[k] = min(updated[k], caps[key])
            if not np.array_equal(updated, state.posted_prices):
                state.posted_prices = updated

    def snapshot(self):
        t = self.trigger
        return TriggerSnapshot(
            super().snapshot(),
            tuple(t.active_until.items()) if t else (),
            tuple(copy.deepcopy(t.history)) if t else (),
            tuple(self._regulator_reports.items()),
            t.last_period if t else None,
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, TriggerSnapshot):
            raise TypeError("trigger snapshot required")
        super().restore(snapshot.native)
        self._regulator_reports = dict(snapshot.reports)
        if self.trigger:
            self.trigger.active_until = dict(snapshot.active_until)
            self.trigger.history = list(copy.deepcopy(snapshot.history))
            self.trigger.last_period = snapshot.last_period

    def branch(self):
        other = TriggerMarketEngine(
            self.compiled,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            trigger_rule=self.trigger.rule if self.trigger else None,
            mechanical_benchmark=self.trigger.benchmark if self.trigger else None,
            seat_reserve=self.seat_reserve,
            rescue_menu=self.rescue_menu,
            rebooking=self.rebooking,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
