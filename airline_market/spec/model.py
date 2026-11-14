"""Frozen specification dataclasses for model Eqs. 1–25.

Field names follow IMPLEMENTATION_PLAN.md §4. Nested mappings are immutable.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Mapping
from ..core.snapshot import deep_freeze
from .schedule import PeriodSchedule


class SpecBase:
    """
    Recursively immutable specification base; validation is performed at load time (Eqs.
    1–19).
    """

    def __post_init__(self):
        """
        Recursively freeze specification fields without reading simulator state (Eqs.
        1–19).
        """
        for f in fields(self):
            object.__setattr__(self, f.name, deep_freeze(getattr(self, f.name)))


@dataclass(frozen=True, kw_only=True)
class HorizonSpec(SpecBase):
    """Booking periods and the cohort departure/collection calendar (Eqs. 1–19)."""

    n_periods: int
    period_days: tuple[float, ...] | None = None
    departure_date: str | None = None
    booking_window_days: float | None = None


@dataclass(frozen=True, kw_only=True)
class AirportSpec(SpecBase):
    """Airport identifier and optional geography/timezone metadata (Eqs. 1–19)."""

    code: str
    name: str = ""
    timezone: str = "UTC"
    lat: float | None = None
    lon: float | None = None


@dataclass(frozen=True, kw_only=True)
class CarrierSpec(SpecBase):
    """Carrier identity and declared model role (Eqs. 1–19)."""

    code: str
    name: str = ""
    kind: str = "network"


@dataclass(frozen=True, kw_only=True)
class CapacityDistribution(SpecBase):
    """Discrete initial leg-seat law and its public/private visibility (Eqs. 1–19)."""

    support: tuple[int, ...]
    probabilities: tuple[float, ...]
    public: bool = False


@dataclass(frozen=True, kw_only=True)
class JointCapacityPrior(SpecBase):
    """Within-carrier correlated leg-capacity support and probabilities (Eqs. 1–19)."""

    legs: tuple[str, ...]
    support: tuple[tuple[int, ...], ...]
    probabilities: tuple[float, ...]
    public: bool = False


@dataclass(frozen=True, kw_only=True)
class LegSpec(SpecBase):
    """
    Dated physical capacity resource and optional operating-cost attributes (Eqs. 1–19).
    """

    id: str
    carrier: str
    origin: str
    dest: str
    dep_time: str | None = None
    arr_time: str | None = None
    capacity: CapacityDistribution
    distance_mi: float | None = None
    cost_per_pax: float = 0.0


@dataclass(frozen=True, kw_only=True)
class ProductSpec(SpecBase):
    """
    Itinerary offer, resource path, price bounds and per-passenger cost (Eqs. 1–19).
    """

    id: str
    carrier: str
    market: str
    legs: tuple[str, ...]
    service_cost: float
    fare_floor: float
    fare_cap: float
    reference_fare: float
    priority: int
    controlled: bool = True
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MarketDefSpec(SpecBase):
    """Directional O&D demand market and competitive-segment membership (Eqs. 1–19)."""

    id: str
    origin: str
    dest: str
    in_segment: bool
    label: str = ""


@dataclass(frozen=True, kw_only=True)
class TasteClass(SpecBase):
    """
    Passenger price sensitivity, airline preference and itinerary utility shifts (Eqs.
    1–19).
    """

    weight: float
    alpha: float
    carrier_pref: Mapping[str, float]
    market_value: Mapping[str, float]
    connection_penalty: float = 0.0
    product_shift: Mapping[str, float] = field(default_factory=dict)
    time_penalty_per_hour: float = 0.0
    schedule_penalty_per_hour: float = 0.0
    ideal_dep_hour: float | None = None


@dataclass(frozen=True, kw_only=True)
class SegmentSpec(SpecBase):
    """
    Passenger segment's finite taste mixture and optional random coefficients (Eqs.
    1–19).
    """

    id: str
    classes: tuple[TasteClass, ...]
    mixed: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class DemandSpec(SpecBase):
    """
    Bernoulli/Poisson arrivals, segment mixtures and exogenous multipliers (Eqs. 1–19).
    """

    mode: str
    arrival_probability: PeriodSchedule | None = None
    arrival_probability_cap: float = 0.98
    market_weights: PeriodSchedule | None = None
    intensity: Mapping[str, PeriodSchedule] | None = None
    segment_weights: PeriodSchedule | Mapping[str, PeriodSchedule]
    holiday_multiplier: float = 1.0
    exogenous_multipliers: Mapping[str, float] = field(default_factory=dict)
    poisson_truncation: int = 3


@dataclass(frozen=True, kw_only=True)
class ChoiceSpec(SpecBase):
    """Utility-model family and connection-penalty convention (Eqs. 1–19)."""

    model: str = "latent_class_mnl"
    connection_penalty_rule: str = "per_connection"
    n_mixed_draws: int = 0


@dataclass(frozen=True, kw_only=True)
class ExoState(SpecBase):
    """Public or latent external condition and its declared attributes (Eqs. 1–19)."""

    name: str
    public: bool = True
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ExogenousSpec(SpecBase):
    """
    Finite condition prior, period transitions and optional public forecasts (Eqs.
    1–19).
    """

    states: tuple[ExoState, ...]
    initial: tuple[float, ...]
    transitions: Mapping[int, tuple[tuple[float, ...], ...]]
    default_transition: tuple[tuple[float, ...], ...] | None = None
    forecasts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CapacityEventSpec(SpecBase):
    """Triggered leg-capacity changes and once-only event semantics (Eqs. 1–19)."""

    trigger: Mapping[str, Any]
    effects: tuple[tuple[str, int], ...]
    once: bool = True


@dataclass(frozen=True, kw_only=True)
class CapacityProcessSpec(SpecBase):
    """Collection of operational capacity events (Eqs. 1–19)."""

    events: tuple[CapacityEventSpec, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ResolutionSpec(SpecBase):
    """
    Whole-itinerary cancellation, refund and service-penalty convention (Eqs. 1–19).
    """

    rule: str = "fixed_priority_cancel_refund"
    refund: str = "full_original_fare"
    penalty_per_cancellation: float = 0.0
    stranded_reentry: bool = False


@dataclass(frozen=True, kw_only=True)
class ShoppingRequestSpec(SpecBase):
    """
    Observer, products, lag, cadence and explicit quote measurement channel (Eqs. 1–19).
    """

    id: str
    observer: str
    target_products: tuple[str, ...]
    lag_periods: int = 0
    every_k_periods: int = 1
    fare_bin: float = 0.0
    fare_noise_sd: float = 0.0
    missing_probability: float = 0.0
    availability: str = "flag"
    seats_cap: int = 9


@dataclass(frozen=True, kw_only=True)
class QuoteSpec(SpecBase):
    """Declared collection of permitted shopping requests (Eqs. 1–19)."""

    requests: tuple[ShoppingRequestSpec, ...]


@dataclass(frozen=True, kw_only=True)
class FeatureSpec(SpecBase):
    """Named pricing feature and its model parameters (Eqs. 1–19)."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CarrierPricingSpec(SpecBase):
    """
    Additive base/fixed/controllable features, coefficient bounds and cadence (Eqs.
    1–19).
    """

    carrier: str
    base: Mapping[str, float]
    time_ramp: float = 0.0
    fixed_features: tuple[tuple[FeatureSpec, float], ...] = ()
    controllable_features: tuple[FeatureSpec, ...] = ()
    theta_bounds: tuple[tuple[float, float], ...] = ()
    theta_grid: tuple[tuple[float, ...], ...] | None = None
    sharing: str = "carrier"
    update_every: int = 1


@dataclass(frozen=True, kw_only=True)
class ControllerAssignment(SpecBase):
    """Carrier-to-controller binding and declared controller parameters (Eqs. 1–19)."""

    carrier: str
    controller: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ObjectiveSpec(SpecBase):
    """Profit normalization and competitive journey-share tradeoff (Eqs. 1–19)."""

    carrier: str
    profit_scale: float
    share_weight: float = 0.0
    share_markets: tuple[str, ...] | None = None
    share_statistic: str = "realized_ratio"


@dataclass(frozen=True, kw_only=True)
class InformationArmSpec(SpecBase):
    """Permitted request subset and explicit oracle access flag (Eqs. 1–19)."""

    name: str
    observer: str
    requests: tuple[str, ...]
    privileged: bool = False
    derived: bool = False


@dataclass(frozen=True, kw_only=True)
class BackgroundRouteSpec(SpecBase):
    """Independent reference supply, quote rule and demand law (Eqs. 1–19)."""

    id: str
    carrier: str
    capacity: CapacityDistribution
    quote_rule: Mapping[str, float]
    purchase: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class TimingSpec(SpecBase):
    """Posting order and initial previous-price convention (Eqs. 1–19)."""

    mode: str
    leaders: tuple[str, ...] = ()
    initial_previous_prices: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MarketSpec(SpecBase):
    """
    Complete immutable network, behavior, control and information specification (Eqs.
    1–19).
    """

    schema_version: str
    name: str
    description: str = ""
    horizon: HorizonSpec
    airports: tuple[AirportSpec, ...]
    carriers: tuple[CarrierSpec, ...]
    legs: tuple[LegSpec, ...]
    capacity_priors: tuple[JointCapacityPrior, ...] = ()
    products: tuple[ProductSpec, ...]
    markets: tuple[MarketDefSpec, ...]
    segments: tuple[SegmentSpec, ...]
    demand: DemandSpec
    choice: ChoiceSpec
    exogenous: ExogenousSpec
    capacity_process: CapacityProcessSpec
    resolution: ResolutionSpec
    quotes: QuoteSpec
    pricing: tuple[CarrierPricingSpec, ...]
    controllers: tuple[ControllerAssignment, ...]
    objectives: tuple[ObjectiveSpec, ...]
    information: tuple[InformationArmSpec, ...]
    background: tuple[BackgroundRouteSpec, ...] = ()
    timing: TimingSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
