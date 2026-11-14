"""A.14/B0/V9: opt-in reserved reaccommodation seats on the native clock.

An integer floor of the declared share of reveal-time remaining seats is held.
The caller supplies the release period; no period-to-24-hour mapping is inferred.
Held-first allocation and capacity-loss writeoffs are explicit ledger events.
"""

import copy
from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal

import numpy as np

from airline_market.policy.rescue import RescueMarketEngine
from airline_market.sim.arrivals import Request
from airline_market.sim.choice import sampled_choice, systematic_utilities
from airline_market.sim.ledger import book, inventory, sellable


@dataclass(frozen=True)
class SeatReserve:
    reveal_period: int
    release_period: int
    share: Decimal
    affected_legs: tuple[str, ...]

    def __post_init__(self):
        if (
            type(self.reveal_period) is not int
            or type(self.release_period) is not int
            or not 0 <= self.reveal_period < self.release_period
        ):
            raise ValueError("explicit ordered native reveal/release periods required")
        share = Decimal(str(self.share))
        if not share.is_finite() or share not in map(
            Decimal, ("0", ".02", ".05", ".10")
        ):
            raise ValueError("reserve share must be 0, 2, 5 or 10 percent")
        legs = tuple(self.affected_legs)
        if (
            not legs
            or len(set(legs)) != len(legs)
            or any(not isinstance(x, str) for x in legs)
        ):
            raise ValueError("distinct explicit affected leg IDs required")
        object.__setattr__(self, "share", share)
        object.__setattr__(self, "affected_legs", legs)


@dataclass(frozen=True)
class ReserveSnapshot:
    native: object
    activated: bool
    released: bool
    ledger: tuple
    events: tuple


class ReserveMarketEngine(RescueMarketEngine):
    """Physical executor only; no old belief or planner kernel is compatible.

    Ordinary demand sees remaining minus held stock. A proved original stranded
    passenger can use held stock at the private regulated fare, with own-carrier
    priority and the existing optional interline rule. Native book/resolve and
    physical capacity are unchanged. Controllers receive their native inputs.
    """

    def __init__(self, compiled, controllers=None, *, seat_reserve=None, **kwargs):
        if seat_reserve is not None and not isinstance(seat_reserve, SeatReserve):
            raise TypeError("explicit SeatReserve required")
        super().__init__(compiled, controllers, **kwargs)
        self.seat_reserve = seat_reserve
        if seat_reserve is not None:
            p = seat_reserve
            if set(p.affected_legs) - set(compiled.leg_index):
                raise ValueError("unknown affected leg")
            if p.release_period > compiled.spec.horizon.n_periods:
                raise ValueError("release outside native horizon")
            if self.rebooking is None or self.rescue_menu is None:
                raise ValueError("explicit rebooking and regulated-fare menu required")
            markets = {
                int(compiled.product_market[j])
                for j, product in enumerate(compiled.spec.products)
                if set(product.legs) & set(p.affected_legs)
            }
            if any(
                self.rescue_menu.fare_for(compiled.spec.markets[m].id, t) is None
                for m in markets
                for t in range(p.reveal_period, p.release_period)
            ):
                raise ValueError(
                    "regulated menu must cover every affected O&D throughout hold"
                )
        self._clear_reserve()

    def _clear_reserve(self):
        self._reserve_activated = False
        self._reserve_released = False
        self._reserve_eligible = False
        self._reserve_ledger = {}
        self.reserve_events = []

    def reset(self, seed=None, *, tape=None):
        self._clear_reserve()
        return super().reset(seed, tape=tape)

    def _remaining(self):
        return {
            self.compiled.spec.legs[l].id: int(x)
            for i, cs in enumerate(self.world.carriers)
            for l, x in zip(self.compiled.legs_of[i], inventory(cs, self.compiled, i))
        }

    def _event(self, leg, kind, count, **extra):
        self.reserve_events.append(
            dict(period=self.world.t, leg=leg, kind=kind, seats=count, **extra)
        )

    def _sync_reserve(self):
        p = self.seat_reserve
        if p is None or self.world is None:
            return
        stock = self._remaining()
        if not self._reserve_activated and self.world.t >= p.reveal_period:
            self._reserve_activated = True
            for leg in p.affected_legs:
                n = int((p.share * stock[leg]).to_integral_value(rounding=ROUND_FLOOR))
                self._reserve_ledger[leg] = {
                    "initial": n,
                    "held": n,
                    "used": 0,
                    "released": 0,
                    "capacity_lost": 0,
                }
                self._event(
                    leg, "hold", n, reveal_remaining=stock[leg], rounding="floor"
                )
        for leg, row in self._reserve_ledger.items():
            lost = max(0, row["held"] - stock[leg])
            if lost:
                row["held"] -= lost
                row["capacity_lost"] += lost
                self._event(leg, "capacity_loss", lost)
        if (
            self._reserve_activated
            and not self._reserve_released
            and self.world.t >= p.release_period
        ):
            self._reserve_released = True
            for leg, row in self._reserve_ledger.items():
                n = row["held"]
                row["released"] += n
                row["held"] = 0
                self._event(leg, "release", n)

    def begin_period(self, actions=None):
        # The parent first applies permanent exits at this boundary. Holds use
        # actual remaining seats, after those native cancellations/refunds.
        result = super().begin_period(actions)
        self._sync_reserve()
        return result

    def _sellable_global(self):
        self._sync_reserve()
        if self.seat_reserve is None or self._reserve_eligible:
            return super()._sellable_global()
        c = self.compiled
        active = np.zeros(len(c.spec.products), dtype=bool)
        for i, cs in enumerate(self.world.carriers):
            free = inventory(cs, c, i).copy()
            for k, l in enumerate(c.legs_of[i]):
                free[k] -= self._reserve_ledger.get(c.spec.legs[l].id, {}).get(
                    "held", 0
                )
            active[list(c.products_of[i])] = sellable(free, c.A[i])
        return active

    def _held_on(self, product):
        return {
            leg: min(1, self._reserve_ledger.get(leg, {}).get("held", 0))
            for leg in self.compiled.spec.products[product].legs
        }

    def _reenter(self, prices):
        if self.seat_reserve is None or self.seat_reserve.share == 0:
            return super()._reenter(prices)
        self._sync_reserve()
        c, t = self.compiled, self.world.t
        purchases = []
        ordered = sorted(
            self._passengers.items(),
            key=lambda pair: (
                pair[1].first_stranded_at,
                pair[1].original.period,
                pair[1].original.arrival_rank,
            ),
        )
        for root, passenger in ordered:
            if (
                passenger.active_record is not None
                or t >= passenger.deadline
                or (root, t) in self._attempts
            ):
                continue
            self._attempts.add((root, t))
            b = passenger.original
            market = int(c.product_market[b.product])
            private_prices, rescue = self._rescue_prices(root, passenger, prices)
            eligible = bool(rescue["rescue_eligible"])
            private_prices = private_prices.copy()
            # R-RESERVE offers the regulated option on held-seat itineraries;
            # a route with no held stock keeps its ordinary/capped fare.
            for j in c.market_products[market]:
                if not any(self._held_on(j).values()):
                    private_prices[j] = prices[j]
            taste, gumbels = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier] + ([None] if self.rebooking.interline else [])
            chosen, stage = None, None
            self._reserve_eligible = eligible
            try:
                for carrier in stages:
                    active = self._sellable_global().copy()
                    for j in c.market_products[market]:
                        active[j] &= (
                            int(c.product_carrier[j]) == carrier
                            if carrier is not None
                            else int(c.product_carrier[j]) != b.carrier
                        )
                    utilities = systematic_utilities(
                        c, market, taste, private_prices, active
                    )
                    selected = sampled_choice(utilities, gumbels[list(columns)])
                    if selected < len(columns) - 1:
                        chosen = columns[selected]
                        stage = "own" if carrier is not None else "interline"
                        break
            finally:
                self._reserve_eligible = False
            record = dict(
                period=t,
                original_record=root,
                original_request=(b.period, b.arrival_rank),
                market=market,
                deadline=passenger.deadline,
                chosen=chosen,
                stage=stage,
                alpha=taste.alpha,
                replacement_record=None,
                fare=None,
                rescue_discount_used=False,
                reserve_eligible=eligible,
                held_seats_used={},
                **rescue,
            )
            if chosen is not None:
                request = Request(-1 - root, market, b.segment, b.taste_class, -1, -1)
                new = book(
                    self.world,
                    int(c.product_carrier[chosen]),
                    chosen,
                    float(private_prices[chosen]),
                    request,
                    c,
                )
                self._replacement_roots[new.record_id] = root
                self._passengers[root] = replace(passenger, active_record=new.record_id)
                used = self._held_on(chosen) if eligible else {}
                for leg, n in used.items():
                    if n:
                        self._reserve_ledger[leg]["held"] -= n
                        self._reserve_ledger[leg]["used"] += n
                        self._event(
                            leg,
                            "eligible_use",
                            n,
                            original_record=root,
                            replacement_record=new.record_id,
                        )
                record.update(
                    replacement_record=new.record_id,
                    fare=new.fare,
                    rescue_discount_used=bool(private_prices[chosen] < prices[chosen]),
                    held_seats_used={leg: n for leg, n in used.items() if n},
                )
                purchases.append(new)
            self.rebooking_history.append(record)
        return tuple(purchases)

    def step(self, actions=None, *, _requests=None, _choices=None):
        p = self.seat_reserve
        if (
            p is not None
            and p.share
            and self.world is not None
            and p.reveal_period <= self.world.t < p.release_period
            and _choices is not None
        ):
            raise ValueError(
                "forced-choice test hook bypasses reserve admission; use native choice"
            )
        result = super().step(actions, _requests=_requests, _choices=_choices)
        self._sync_reserve()
        self.reserve_balance()
        return result

    def reserve_balance(self):
        self._sync_reserve()
        stock = self._remaining()
        result = []
        for leg, row in self._reserve_ledger.items():
            assert (
                row["initial"]
                == row["held"] + row["used"] + row["released"] + row["capacity_lost"]
            )
            assert 0 <= row["held"] <= stock[leg]
            result.append(
                dict(
                    leg=leg,
                    **row,
                    physical_remaining=stock[leg],
                    ordinary_available=stock[leg] - row["held"],
                )
            )
        return tuple(result)

    def snapshot(self):
        return ReserveSnapshot(
            super().snapshot(),
            self._reserve_activated,
            self._reserve_released,
            tuple(
                (leg, tuple(row.items())) for leg, row in self._reserve_ledger.items()
            ),
            tuple(copy.deepcopy(self.reserve_events)),
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, ReserveSnapshot):
            raise TypeError("reserve snapshot required")
        super().restore(snapshot.native)
        self._reserve_activated = snapshot.activated
        self._reserve_released = snapshot.released
        self._reserve_eligible = False
        self._reserve_ledger = {leg: dict(row) for leg, row in snapshot.ledger}
        self.reserve_events = list(copy.deepcopy(snapshot.events))

    def branch(self):
        other = ReserveMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            seat_reserve=self.seat_reserve,
            rescue_menu=self.rescue_menu,
            rebooking=self.rebooking,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
