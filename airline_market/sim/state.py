"""Booking ledgers and mutable operational state (Eqs. 12–15)."""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Hashable
import numpy as np
from ..core.types import PeriodPhase


@dataclass(frozen=True)
class Booking:
    """Immutable itinerary reservation consuming each leg on its path (Eqs. 3–16)."""

    record_id: int
    carrier: int
    product: int
    fare: float
    period: int
    arrival_rank: int
    segment: int = -1
    taste_class: int = -1


@dataclass
class CarrierState:
    """
    Mutable capacity, active ledger, money and pricing/controller memory (Eqs. 3–16).
    """

    capacity: np.ndarray
    bookings: list[Booking]
    cash: float
    refunds: float
    penalties: float
    canceled: int
    posted_prices: np.ndarray
    previous_prices: np.ndarray
    theta: tuple[float, ...] | None = None
    controller_memory: Hashable = ()
    revenue: float = 0.0

    def clone(self):
        """
        Clone mutable state while sharing immutable model data and random tapes (Eqs.
        3–16).
        """
        return CarrierState(
            self.capacity.copy(),
            list(self.bookings),
            self.cash,
            self.refunds,
            self.penalties,
            self.canceled,
            self.posted_prices.copy(),
            self.previous_prices.copy(),
            self.theta,
            copy.deepcopy(self.controller_memory),
            self.revenue,
        )


@dataclass
class BackgroundState:
    """
    Independent reference-route stock and revenue, outside competitive share (Eqs.
    3–16).
    """

    stock: int
    revenue: float = 0.0


@dataclass
class WorldState:
    """Mutable cohort state kept private from public-information actors (Eqs. 3–16)."""

    t: int
    phase: PeriodPhase
    exo: int
    carriers: list[CarrierState]
    background: list[BackgroundState]
    quote_history: object
    next_record_id: int = 0
    events_fired: set[int] = field(default_factory=set)
    request_log: list[dict] = field(default_factory=list)
    sellout_period: dict[int, int] = field(default_factory=dict)
    initial_capacity: tuple[int, ...] = ()
    last_purchases: tuple[Booking, ...] = ()
    posted_leaders: set[int] = field(default_factory=set)

    def clone(self):
        """
        Clone mutable state while sharing immutable model data and random tapes (Eqs.
        3–16).
        """
        return WorldState(
            self.t,
            self.phase,
            self.exo,
            [c.clone() for c in self.carriers],
            [copy.copy(b) for b in self.background],
            self.quote_history.clone(),
            self.next_record_id,
            set(self.events_fired),
            copy.deepcopy(self.request_log),
            dict(self.sellout_period),
            self.initial_capacity,
            self.last_purchases,
            set(self.posted_leaders),
        )
