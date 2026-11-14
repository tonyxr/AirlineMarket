"""Finite request model with exact choice probabilities and explicit seats.

A controls OH and connecting OD. B also sells feeder HD. The feeder and
connection share B's HD resource. An independent background XY route provides
the unrelated-quote control and is outside the evaluated departure segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import product
import math

import numpy as np


class Information(str, Enum):
    DIRECT = "direct"
    LINKED = "linked"
    UNRELATED = "unrelated"
    ORACLE = "oracle"


@dataclass(frozen=True)
class Config:
    horizon: int = 10
    own_capacity: tuple[int, int] = (2, 2)
    rival_oh_capacity: int = 2
    rival_hd_capacities: tuple[int, ...] = (1, 3)
    arrival_probability: float = 0.9
    holiday_multiplier: float = 1.0
    capacity_shock_probability: float = 0.0
    shock_step: int = 5
    disruption_penalty: float = 25.0
    feeder_premium: float = 35.0
    rival_response: float = 0.10
    share_weight: float = 0.0
    profit_scale: float = 300.0

    def __post_init__(self) -> None:
        if self.horizon < 2 or not 0 < self.shock_step < self.horizon:
            raise ValueError("horizon >= 2 and 0 < shock_step < horizon required")
        capacities = self.own_capacity + (self.rival_oh_capacity,) + self.rival_hd_capacities
        if not self.rival_hd_capacities or any(type(x) is not int or x < 1 for x in capacities):
            raise ValueError("initial capacities must be positive integers")
        if len(self.own_capacity) != 2 or len(set(self.rival_hd_capacities)) != len(self.rival_hd_capacities):
            raise ValueError("two own legs and distinct rival HD capacity states required")
        values = (self.arrival_probability, self.holiday_multiplier,
                  self.capacity_shock_probability, self.disruption_penalty,
                  self.feeder_premium, self.rival_response, self.share_weight,
                  self.profit_scale)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("configuration numbers must be finite")
        if not 0 <= self.arrival_probability <= 1 or not 0 <= self.capacity_shock_probability <= 1:
            raise ValueError("probabilities must be in [0, 1]")
        if self.holiday_multiplier <= 0 or self.profit_scale <= 0:
            raise ValueError("holiday multiplier and profit scale must be positive")
        if min(self.disruption_penalty, self.feeder_premium, self.rival_response, self.share_weight) < 0:
            raise ValueError("costs, premiums, responses and objective weights must be nonnegative")


# Indices are local to each carrier; connecting bookings consume BOTH legs.
INCIDENCE = (((1, 0), (1, 1)), ((1, 0), (0, 1), (1, 1)))
MARKETS = (("OH", "OD"), ("OH", "HD", "OD"))
COSTS = ((15.0, 30.0), (15.0, 12.0, 30.0))
COEFFICIENT_GRID = tuple(product((-30.0, 0.0, 30.0), (-30.0, 0.0, 30.0), (0.0, 15.0)))
# (price sensitivity, carrier A preference, carrier B preference, OH value, OD value)
TASTES = ((0.025, 0.0, 0.30, 4.0, 5.0), (0.012, 0.25, 0.0, 3.5, 4.5))


@dataclass(frozen=True, order=True)
class RivalState:
    capacity: tuple[int, int]
    sold: tuple[int, int, int] = (0, 0, 0)

    @property
    def inventory(self) -> tuple[int, int]:
        return (self.capacity[0] - self.sold[0] - self.sold[2],
                self.capacity[1] - self.sold[1] - self.sold[2])

    @property
    def segment_journeys(self) -> int:
        return self.sold[0] + self.sold[2]

    def purchase(self, j: int) -> RivalState:
        sold = list(self.sold)
        sold[j] += 1
        result = RivalState(self.capacity, tuple(sold))
        if min(result.inventory) < 0:
            raise ValueError("rival purchase exceeds leg capacity")
        return result

    def capacity_loss(self) -> RivalState:
        capacity = (self.capacity[0], max(0, self.capacity[1] - 1))
        remaining = list(capacity)
        kept = [0, 0, 0]
        # The simulator uses the same fixed product-priority cancellation rule.
        for j in range(3):
            for _ in range(self.sold[j]):
                use = INCIDENCE[1][j]
                if all(u <= x for u, x in zip(use, remaining)):
                    kept[j] += 1
                    remaining = [x - u for x, u in zip(remaining, use)]
        return RivalState(capacity, tuple(kept))


@dataclass(frozen=True)
class Booking:
    product: int
    fare: float


@dataclass(frozen=True)
class Quote:
    product: str
    fare: float
    available: bool


@dataclass(frozen=True)
class Observation:
    time: int
    own_capacity: tuple[int, int]
    own_bookings: tuple[Booking, ...]
    own_cash: float
    previous_own_prices: tuple[float, float]
    weather_event: bool
    quotes: tuple[Quote, ...]
    privileged_rival: RivalState | None = None

    @property
    def own_inventory(self) -> tuple[int, int]:
        return inventory(0, self.own_capacity, self.own_bookings)


@dataclass
class WorldState:
    time: int
    capacity: list[tuple[int, int]]
    bookings: list[list[Booking]] = field(default_factory=lambda: [[], []])
    cash: list[float] = field(default_factory=lambda: [0.0, 0.0])
    previous_own_prices: tuple[float, float] = (110.0, 185.0)
    weather_event: bool = False
    reference_inventory: int = 3
    refunds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    canceled: list[int] = field(default_factory=lambda: [0, 0])

    def clone(self) -> WorldState:
        # Booking records and capacity tuples are immutable; copy mutable containers.
        return WorldState(self.time, list(self.capacity), [list(b) for b in self.bookings],
                          list(self.cash), self.previous_own_prices, self.weather_event,
                          self.reference_inventory, list(self.refunds), list(self.canceled))

    def rival(self) -> RivalState:
        return RivalState(self.capacity[1], tuple(sum(b.product == j for b in self.bookings[1]) for j in range(3)))


@dataclass(frozen=True)
class NoiseTape:
    """All exogenous draws are indexed before any policy-dependent branching."""
    requests: np.ndarray
    gumbels: np.ndarray
    reference: np.ndarray
    weather_uniform: float
    initial_uniform: float
    reference_initial: int

    @classmethod
    def from_seed(cls, seed: int, horizon: int) -> NoiseTape:
        children = np.random.SeedSequence(seed).spawn(5)
        requests = np.random.default_rng(children[0]).random((horizon, 3))
        gumbels = np.random.default_rng(children[1]).gumbel(size=(horizon, 6))
        ref_rng = np.random.default_rng(children[2])
        reference = ref_rng.random(horizon)
        ref_initial = int(ref_rng.choice((1, 3)))
        weather = float(np.random.default_rng(children[3]).random())
        initial = float(np.random.default_rng(children[4]).random())
        for array in (requests, gumbels, reference):
            array.setflags(write=False)
        return cls(requests, gumbels, reference, weather, initial, ref_initial)


def inventory(carrier: int, capacity: tuple[int, int], bookings) -> tuple[int, int]:
    return tuple(capacity[l] - sum(INCIDENCE[carrier][b.product][l] for b in bookings) for l in range(2))


def available(carrier: int, j: int, stock: tuple[int, int]) -> bool:
    return all(use <= x for use, x in zip(INCIDENCE[carrier][j], stock))


def categorical(u: float, weights) -> int:
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if u < cumulative:
            return index
    return len(weights) - 1


class Model:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def prior(self) -> dict[RivalState, float]:
        values = self.config.rival_hd_capacities
        return {RivalState((self.config.rival_oh_capacity, h)): 1 / len(values) for h in values}

    def arrival_probability(self, weather_event: bool) -> float:
        return min(0.98, self.config.arrival_probability * self.config.holiday_multiplier * (0.70 if weather_event else 1.0))

    def market_weights(self, time: int) -> tuple[float, float, float]:
        return (0.55, 0.40, 0.05) if time < self.config.horizon * 0.4 else (0.15, 0.15, 0.70)

    def taste_weights(self, time: int) -> tuple[float, float]:
        business = 0.20 if time < self.config.horizon * 0.4 else 0.70
        return (1 - business, business)

    def rival_quotes(self, rival: RivalState, previous_own_prices) -> tuple[Quote, ...]:
        x = rival.inventory
        rho = self.config.rival_response
        prices = (125.0 + 15.0 * (x[0] <= 1) + rho * (previous_own_prices[0] - 110),
                  85.0 + self.config.feeder_premium * (x[1] <= 1),
                  170.0 + rho * (previous_own_prices[1] - 185))
        return tuple(Quote(MARKETS[1][j], round(float(p), 2), available(1, j, x)) for j, p in enumerate(prices))

    @staticmethod
    def reference_quote(stock: int) -> Quote:
        return Quote("XY", 85.0 + 35.0 * (stock <= 1), stock > 0)

    def prices(self, theta, time: int, own_stock, direct_quotes) -> tuple[float, float]:
        if len(theta) != 3 or not all(math.isfinite(float(x)) for x in theta):
            raise ValueError("three finite pricing coefficients required")
        quotes = {q.product: q for q in direct_quotes}
        prices = []
        for j, (base, rival_base) in enumerate(((110.0, 125.0), (185.0, 170.0))):
            q = quotes[MARKETS[0][j]]
            gap = q.fare - rival_base if q.available else 0.0
            scarcity = sum(u / (1 + x) for u, x in zip(INCIDENCE[0][j], own_stock))
            adjustment = theta[0] + (theta[1] if j == 1 else 0.0) + scarcity * theta[2]
            price = base + 15 * time / (self.config.horizon - 1) + 0.10 * gap + adjustment
            prices.append(round(float(min(300.0, max(70.0, price))), 2))
        return tuple(prices)

    @staticmethod
    def utilities(market: str, taste: int, own_stock, rival_stock, own_prices, rival_prices) -> np.ndarray:
        alpha, pref_a, pref_b, oh_value, od_value = TASTES[taste]
        value = od_value if market == "OD" else oh_value
        result = np.full(6, -np.inf)
        result[5] = 0.0  # Outside option.
        for carrier in (0, 1):
            stock = own_stock if carrier == 0 else rival_stock
            prices = own_prices if carrier == 0 else rival_prices
            pref = pref_a if carrier == 0 else pref_b
            for j, name in enumerate(MARKETS[carrier]):
                if name == market and available(carrier, j, stock):
                    idx = j if carrier == 0 else j + 2
                    result[idx] = value + pref - alpha * prices[j] - (0.5 if name == "OD" else 0.0)
        return result

    @staticmethod
    def probabilities(utilities: np.ndarray) -> np.ndarray:
        exp = np.exp(utilities - np.max(utilities))
        return exp / exp.sum()

    @lru_cache(maxsize=100_000)
    def outcomes(self, time: int, own_stock: tuple[int, int], rival_stock: tuple[int, int],
                 own_prices: tuple[float, float], previous_own_prices: tuple[float, float],
                 weather_event: bool) -> tuple[float, ...]:
        # Own OH/OD, rival OH/HD/OD, no own-or-rival purchase.
        arrival = self.arrival_probability(weather_event)
        probs = np.zeros(6)
        probs[5] = 1 - arrival
        # Only inventory enters B's fixed posted-price rule.
        r = RivalState(rival_stock)
        rp = tuple(q.fare for q in self.rival_quotes(r, previous_own_prices))
        for market, wm in zip(("OH", "HD", "OD"), self.market_weights(time)):
            for taste, wt in enumerate(self.taste_weights(time)):
                utilities = self.utilities(market, taste, own_stock, rival_stock, own_prices, rp)
                probs += arrival * wm * wt * self.probabilities(utilities)
        return tuple(float(p) for p in probs)

    def observation(self, state: WorldState, information: Information) -> Observation:
        full = self.rival_quotes(state.rival(), state.previous_own_prices)
        quotes = (full[0], full[2])
        if information == Information.LINKED:
            quotes += (full[1],)
        elif information == Information.UNRELATED:
            quotes += (self.reference_quote(state.reference_inventory),)
        return Observation(state.time, state.capacity[0], tuple(state.bookings[0]), state.cash[0],
                           state.previous_own_prices, state.weather_event, quotes,
                           state.rival() if information == Information.ORACLE else None)

    def planning_state(self, observation: Observation, rival: RivalState) -> WorldState:
        # Reconstruct from observed own state + a posterior sample, never true hidden state.
        rival_bookings = [Booking(j, (125.0, 85.0, 170.0)[j]) for j in range(3) for _ in range(rival.sold[j])]
        return WorldState(observation.time, [observation.own_capacity, rival.capacity],
                          [list(observation.own_bookings), rival_bookings],
                          [observation.own_cash, sum(b.fare for b in rival_bookings)],
                          observation.previous_own_prices, observation.weather_event)

    def resolve(self, state: WorldState, carrier: int) -> None:
        left = list(state.capacity[carrier])
        kept = []
        for booking in sorted(state.bookings[carrier], key=lambda b: b.product):
            use = INCIDENCE[carrier][booking.product]
            if all(u <= x for u, x in zip(use, left)):
                kept.append(booking)
                left = [x - u for x, u in zip(left, use)]
            else:
                state.cash[carrier] -= booking.fare + self.config.disruption_penalty
                state.refunds[carrier] += booking.fare
                state.canceled[carrier] += 1
        state.bookings[carrier] = kept

    def step(self, state: WorldState, theta, tape: NoiseTape) -> int | None:
        t = state.time
        if t >= self.config.horizon:
            raise ValueError("cannot step a terminal state")
        own_stock = inventory(0, state.capacity[0], state.bookings[0])
        rq = self.rival_quotes(state.rival(), state.previous_own_prices)
        prices = self.prices(theta, t, own_stock, (rq[0], rq[2]))
        own_sale = None
        if tape.requests[t, 0] < self.arrival_probability(state.weather_event):
            market = ("OH", "HD", "OD")[categorical(tape.requests[t, 1], self.market_weights(t))]
            taste = categorical(tape.requests[t, 2], self.taste_weights(t))
            utilities = self.utilities(market, taste, own_stock, state.rival().inventory,
                                       prices, tuple(q.fare for q in rq))
            selected = int(np.argmax(utilities + tape.gumbels[t]))
            if selected < 5:
                carrier, j = (0, selected) if selected < 2 else (1, selected - 2)
                fare = prices[j] if carrier == 0 else rq[j].fare
                state.bookings[carrier].append(Booking(j, fare))
                state.cash[carrier] += fare
                if carrier == 0:
                    own_sale = j
        # Independent resource, demand stream, and quote process for the placebo route.
        ref = self.reference_quote(state.reference_inventory)
        ref_buy = 0.7 / (1 + math.exp(-(4 - 0.025 * ref.fare)))
        if ref.available and tape.reference[t] < ref_buy:
            state.reference_inventory -= 1
        state.previous_own_prices = prices
        state.time += 1
        if state.time == self.config.shock_step and tape.weather_uniform < self.config.capacity_shock_probability:
            state.weather_event = True
            state.capacity[0] = (max(0, state.capacity[0][0] - 1), state.capacity[0][1])
            state.capacity[1] = (state.capacity[1][0], max(0, state.capacity[1][1] - 1))
            for carrier in (0, 1):
                self.resolve(state, carrier)
        for carrier in (0, 1):
            if min(inventory(carrier, state.capacity[carrier], state.bookings[carrier])) < 0:
                raise AssertionError("shared-leg capacity conservation failed")
        return own_sale

    def metrics(self, state: WorldState) -> dict[str, float]:
        profits = [state.cash[i] - sum(COSTS[i][b.product] for b in state.bookings[i]) for i in (0, 1)]
        own = len(state.bookings[0])
        rival_segment = state.rival().segment_journeys
        share = own / (own + rival_segment) if own + rival_segment else 0.0
        return {"profit": profits[0], "rival_profit": profits[1], "share": share,
                "objective": profits[0] / self.config.profit_scale + self.config.share_weight * share,
                "journeys": float(own), "rival_segment_journeys": float(rival_segment),
                "refunds": state.refunds[0], "cancellations": float(state.canceled[0])}


class AirlineEnv:
    def __init__(self, model: Model | None = None):
        self.model = model or Model()
        self.state: WorldState | None = None
        self.tape: NoiseTape | None = None

    def reset(self, seed: int) -> None:
        c = self.model.config
        self.tape = NoiseTape.from_seed(seed, c.horizon)
        h = c.rival_hd_capacities[categorical(self.tape.initial_uniform, [1 / len(c.rival_hd_capacities)] * len(c.rival_hd_capacities))]
        self.state = WorldState(0, [c.own_capacity, (c.rival_oh_capacity, h)], reference_inventory=self.tape.reference_initial)

    def observe(self, information: Information) -> Observation:
        if self.state is None:
            raise RuntimeError("reset the environment first")
        return self.model.observation(self.state, information)

    def step(self, theta) -> int | None:
        if self.state is None or self.tape is None:
            raise RuntimeError("reset the environment first")
        return self.model.step(self.state, theta, self.tape)

    def snapshot(self) -> WorldState:
        if self.state is None:
            raise RuntimeError("reset the environment first")
        return self.state.clone()

    def restore(self, snapshot: WorldState) -> None:
        if self.tape is None:
            raise RuntimeError("restore requires the original episode tape")
        self.state = snapshot.clone()
