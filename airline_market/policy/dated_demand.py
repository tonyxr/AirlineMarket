"""A.5/B0/B1/V9: dated Poisson demand surge and hurricane composition.

Pure declared-date/rate/mixture projection, then frozen shared-tape and physical
execution. No B1 grid, empirical calibration, lifecycle welfare or planner claim.
"""

import copy
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from airline_market.core.hashing import spec_hash
from airline_market.policy.dated_reserve import utc
from airline_market.policy.dated_shocks import (
    B1DatedShock,
    DatedShockMarketEngine,
    _public_dates,
    compile_dated_shocks,
)
from airline_market.policy.shared_demand import shared_poisson_bank
from airline_market.sim.tape import segment_weights
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    try:
        x = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("explicit finite decimal declaration required") from error
    if not x.is_finite():
        raise ValueError("explicit finite decimal declaration required")
    return Fraction(x)


def _rational(x):
    return {"numerator": str(x.numerator), "denominator": str(x.denominator)}


@dataclass(frozen=True)
class DatedDemandSurge:
    """B1 +30/+100 percent; duration and full post-mixture are caller declarations.

    B1 does not supply a demand duration or mix vector. Neither is defaulted from
    weather duration, segment names, historical fares or an invented third level.
    """

    shock_id: str
    reference_leg: str
    reveal_days: int
    surge: object
    stop_utc: object
    markets: tuple
    post_segment_weights: tuple

    def __post_init__(self):
        if not isinstance(self.shock_id, str) or not self.shock_id.strip():
            raise ValueError("explicit shock ID required")
        if not isinstance(self.reference_leg, str) or not self.reference_leg:
            raise ValueError("explicit public reference leg required")
        if type(self.reveal_days) is not int or self.reveal_days not in {14, 7, 3, 1}:
            raise ValueError("B1 reveal days must be 14, 7, 3 or 1")
        surge = _fraction(self.surge)
        if surge not in {Fraction(3, 10), Fraction(1)}:
            raise ValueError("B1 demand surge must be +30% or +100%")
        markets = tuple(sorted(self.markets))
        if (
            not markets
            or len(set(markets)) != len(markets)
            or any(not isinstance(m, str) or not m for m in markets)
        ):
            raise ValueError("distinct explicit affected demand markets required")
        weights = tuple(sorted((k, _fraction(w)) for k, w in self.post_segment_weights))
        if (
            not weights
            or len(dict(weights)) != len(weights)
            or any(not isinstance(k, str) or not k or w < 0 for k, w in weights)
            or sum((w for _, w in weights), Fraction()) != 1
        ):
            raise ValueError("full nonnegative segment mixture must sum exactly to one")
        object.__setattr__(self, "surge", surge)
        object.__setattr__(self, "stop_utc", utc(self.stop_utc))
        object.__setattr__(self, "markets", markets)
        object.__setattr__(self, "post_segment_weights", weights)

    def record(self):
        return {
            "shock_id": self.shock_id,
            "reference_leg": self.reference_leg,
            "reveal_days": self.reveal_days,
            "surge": _rational(self.surge),
            "stop_utc": self.stop_utc.isoformat(),
            "markets": list(self.markets),
            "post_segment_weights": {
                k: _rational(w) for k, w in self.post_segment_weights
            },
        }


def _less_sensitive_witness(compiled, before, after):
    """Nominal finite-alpha FOSD using exact canonical decimal declarations.

    No segment-name proxy, mean-only comparison, weight normalization or
    continuous-mixture approximation. Native execution still uses Float64 fields.
    """
    if compiled.spec.choice.model != "latent_class_mnl" or any(
        s.mixed is not None for s in compiled.spec.segments
    ):
        raise ValueError("declared finite latent classes required for mixture ordering")
    if sum(before, Fraction()) != 1:
        raise ValueError("baseline canonical segment weights must sum exactly to one")
    old, new = {}, {}
    for g, seg in enumerate(compiled.spec.segments):
        weights = tuple(_fraction(c.weight) for c in seg.classes)
        if sum(weights, Fraction()) != 1:
            raise ValueError("canonical finite-class weights must sum exactly to one")
        for cls, w in zip(seg.classes, weights):
            alpha = _fraction(cls.alpha)
            old[alpha] = old.get(alpha, Fraction()) + before[g] * w
            new[alpha] = new.get(alpha, Fraction()) + after[g] * w
    support = sorted(set(old) | set(new))
    a = b = Fraction()
    rows = []
    for alpha in support:
        a += old.get(alpha, Fraction())
        b += new.get(alpha, Fraction())
        rows.append((alpha, a, b))
    if any(b < a for _, a, b in rows) or not any(b > a for _, a, b in rows):
        raise ValueError(
            "post-mixture must strictly shift finite alpha distribution lower"
        )
    return tuple(rows)


@dataclass(frozen=True)
class DatedDemandScenario:
    scenario_id: str
    compiled: object
    base_spec_sha256: str
    clock: object
    departures: tuple
    design: object
    reveal: object
    active_periods: tuple
    mixture_witnesses: tuple
    physical_shocks: tuple = ()
    kind: str = "demand_surge"

    def record(self):
        return {
            "plan_id": "A.5/B0/B1/V9",
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "base_spec_sha256": self.base_spec_sha256,
            "compiled_spec_sha256": self.compiled.spec_hash,
            "clock_utc": [x.isoformat() for x in self.clock.boundaries],
            "leg_departures_utc": {k: v.isoformat() for k, v in self.departures},
            "design": None if self.design is None else self.design.record(),
            "reveal_utc": None if self.reveal is None else self.reveal.isoformat(),
            "active_periods": list(self.active_periods),
            "demand_state": "control"
            if self.design is None
            else "mapped_active_window"
            if self.active_periods
            else "unstarted_in_cohort",
            "physical_shocks": [s.record() for s in self.physical_shocks],
            "mixture_witnesses": [
                {
                    "period": t,
                    "market": m,
                    "finite_alpha_cdf": [
                        {
                            "alpha": _rational(x),
                            "baseline_cdf": _rational(a),
                            "post_cdf": _rational(b),
                        }
                        for x, a, b in rows
                    ],
                }
                for t, m, rows in self.mixture_witnesses
            ],
            "ordering_scope": "Nominal finite canonical-decimal alpha FOSD; native Float64 execution unchanged",
            "day_definition": "24 elapsed UTC hours; native period demand rates are not per-hour exposure",
            "full_B1_or_empirical_adoption": False,
        }

    @property
    def contract_hash(self):
        return spec_hash(self.record())


def compile_dated_demand(
    compiled, clock, departures, design=None, *, control_id="control"
):
    """Read-only public-date/spec projection, no tape, actor, world or RNG access."""
    dep = _public_dates(compiled, clock, departures)
    if design is None:
        return DatedDemandScenario(
            control_id,
            compiled,
            compiled.spec_hash,
            clock,
            dep,
            None,
            None,
            (),
            (),
            kind="control",
        )
    if type(design) is not DatedDemandSurge:
        raise TypeError("DatedDemandSurge required")
    s = compiled.spec
    if s.demand.mode != "poisson":
        raise ValueError(
            "native Poisson demand required; Bernoulli conversion is not inferred"
        )
    if design.reference_leg not in dict(dep) or set(design.markets) - set(
        compiled.market_index
    ):
        raise ValueError("unknown public reference leg or affected demand market")
    weights = dict(design.post_segment_weights)
    if set(weights) != {seg.id for seg in s.segments}:
        raise ValueError("post-mixture must name EVERY compiled segment")
    after = tuple(weights[seg.id] for seg in s.segments)
    reveal = dict(dep)[design.reference_leg] - timedelta(days=design.reveal_days)
    end = design.stop_utc
    if end <= reveal:
        raise ValueError("explicit demand stop must follow reveal")
    if reveal < clock.at(0):
        raise ValueError("pre-cohort demand reveal is unsupported")
    T = s.horizon.n_periods
    if reveal < clock.at(T):
        clock.index(reveal)
    if clock.at(0) < end <= clock.at(T):
        clock.index(end)
    periods = tuple(t for t in range(T) if reveal <= clock.at(t) < end)
    if not periods:
        return DatedDemandScenario(
            design.shock_id,
            compiled,
            compiled.spec_hash,
            clock,
            dep,
            design,
            reveal,
            (),
            (),
        )
    intensity = dict(s.demand.intensity)
    mixtures = {
        m.id: s.demand.segment_weights
        if hasattr(s.demand.segment_weights, "value_at")
        else s.demand.segment_weights[m.id]
        for m in s.markets
    }
    witnesses = []
    for j, market in enumerate(s.markets):
        if market.id not in design.markets:
            continue
        rates, mix = [], []
        for t in range(T):
            native_mix = segment_weights(compiled, t, j)
            if t in periods:
                before = tuple(_fraction(w) for w in native_mix)
                rows = _less_sensitive_witness(compiled, before, after)
                witnesses.append((t, market.id, rows))
                rates.append(
                    float(
                        _fraction(intensity[market.id].value_at(t)) * (1 + design.surge)
                    )
                )
                mix.append(tuple(float(w) for w in after))
            else:
                rates.append(intensity[market.id].value_at(t))
                mix.append(tuple(native_mix))
        intensity[market.id] = PeriodSchedule(tuple(range(T)), tuple(rates))
        mixtures[market.id] = PeriodSchedule(tuple(range(T)), tuple(mix))
    demand = replace(s.demand, intensity=intensity, segment_weights=mixtures)
    changed = compile_market(replace(s, demand=demand))
    return DatedDemandScenario(
        design.shock_id,
        changed,
        compiled.spec_hash,
        clock,
        dep,
        design,
        reveal,
        periods,
        tuple(witnesses),
    )


def compile_dated_hurricane(
    compiled, clock, departures, *, scenario_id, capacity, demand
):
    """Explicit weather + surge + native re-entry; separate declared stop dates."""
    if type(capacity) is not B1DatedShock or capacity.kind != "weather":
        raise ValueError(
            "hurricane capacity component must be declared both-carrier weather"
        )
    if type(demand) is not DatedDemandSurge or (
        capacity.reference_leg != demand.reference_leg
        or capacity.reveal_days != demand.reveal_days
    ):
        raise ValueError("hurricane components need the same explicit reference reveal")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("explicit composite scenario ID required")
    departures = _public_dates(compiled, clock, departures)
    compiled_physical = compile_dated_shocks(compiled, clock, departures, (capacity,))
    out = compile_dated_demand(compiled, clock, departures, demand)
    assert compiled_physical[0].reveal == out.reveal
    return replace(
        out, scenario_id=scenario_id, physical_shocks=(capacity,), kind="hurricane"
    )


@dataclass(frozen=True)
class DatedDemandSnapshot:
    native: object
    scenario_contract: str


class DatedDemandMarketEngine(DatedShockMarketEngine):
    """Bind compiled demand and physical dates, delegate one native lifecycle."""

    def __init__(self, scenario, controllers=None, **kwargs):
        if type(scenario) is not DatedDemandScenario:
            raise TypeError("compiled DatedDemandScenario required")
        if scenario.kind == "hurricane" and kwargs.get("dated_rebooking") is None:
            raise ValueError(
                "hurricane execution requires explicit dated stranded re-entry"
            )
        self.demand_scenario = scenario
        super().__init__(
            scenario.compiled,
            controllers,
            shock_clock=scenario.clock,
            shock_leg_departures=scenario.departures,
            dated_shocks=scenario.physical_shocks,
            **kwargs,
        )

    def snapshot(self):
        return DatedDemandSnapshot(
            super().snapshot(), self.demand_scenario.contract_hash
        )

    def restore(self, snapshot):
        if type(snapshot) is not DatedDemandSnapshot:
            raise TypeError("dated-demand combined snapshot required")
        if snapshot.scenario_contract != self.demand_scenario.contract_hash:
            raise ValueError("snapshot demand/date/composite contract differs")
        super().restore(snapshot.native)

    def branch(self):
        other = DatedDemandMarketEngine(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            ordinary_priority=self.ordinary_priority,
            trigger_rule=self.trigger.rule if self.trigger else None,
            mechanical_benchmark=self.trigger.benchmark if self.trigger else None,
            dated_rebooking=self.dated_rebooking,
            dated_menu=self.dated_menu,
            dated_reserve=self.dated_reserve,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other


@dataclass(frozen=True)
class DatedDemandFamily:
    scenarios: tuple
    bank: object

    def engine(self, name, controllers=None, **kwargs):
        scenario = dict(self.scenarios)[name]
        engine = DatedDemandMarketEngine(scenario, controllers, **kwargs)
        engine.reset(tape=self.bank.project(name))
        return engine

    def record(self):
        return {
            "plan_id": "A.5/B0/B1/V9",
            "shared_bank": self.bank.record(),
            "dated_scenarios": {k: s.record() for k, s in self.scenarios},
        }


def dated_demand_family(scenarios, *, seed, namespace):
    """One new named shared family; no old-bank/legacy-byte identity is claimed."""
    items = tuple(sorted(scenarios.items()))
    if not items or any(type(s) is not DatedDemandScenario for _, s in items):
        raise TypeError("nonempty named compiled dated scenario family required")
    contracts = {(s.base_spec_sha256, s.clock, s.departures) for _, s in items}
    if len(contracts) != 1:
        raise ValueError("dated family must share base spec, UTC clock and departures")
    bank = shared_poisson_bank(
        {k: s.compiled for k, s in items}, seed=seed, namespace=namespace
    )
    return DatedDemandFamily(items, bank)
