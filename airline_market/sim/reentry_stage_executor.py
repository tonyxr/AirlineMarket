"""A.10/A.5/A.14/B0/V9: NEW opt-in native executor revision with stage hooks.

Actual prospective execution, not observer replay. Three frozen native re-entry
bodies adapted ONLY at final same-call hook sites and explicit route delegation.
Source ranges/hashes and stripped-AST agreement are retained locally.
"""

from dataclasses import replace

from airline_market.policy.reserve import ReserveMarketEngine
from airline_market.sim.arrivals import Request
from airline_market.sim.choice import sampled_choice, systematic_utilities
from airline_market.sim.ledger import book


class _StageReentryExecutor(ReserveMarketEngine):
    def _stage_base_reenter(self, prices):
        if self.rebooking is None:
            return ()
        (c, t) = (self.compiled, self.world.t)
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
            if passenger.active_record is not None or t >= passenger.deadline:
                continue
            if (root, t) in self._attempts:
                continue
            self._attempts.add((root, t))
            b = passenger.original
            market = int(c.product_market[b.product])
            (taste, gumbels) = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier]
            if self.rebooking.interline:
                stages.append(None)
            (chosen, stage) = (None, None)
            for carrier in stages:
                active = self._sellable_global().copy()
                for j in c.market_products[market]:
                    active[j] &= (
                        int(c.product_carrier[j]) == carrier
                        if carrier is not None
                        else int(c.product_carrier[j]) != b.carrier
                    )
                self._capture_stage_input(
                    root,
                    t,
                    market,
                    "own" if carrier is not None else "interline",
                    "base",
                    prices,
                    active,
                )
                utilities = systematic_utilities(c, market, taste, prices, active)
                selected = sampled_choice(utilities, gumbels[list(columns)])
                self._capture_stage_result(root, t, columns, selected)
                if selected < len(columns) - 1:
                    chosen = columns[selected]
                    stage = "own" if carrier is not None else "interline"
                    break
            record = {
                "period": t,
                "original_record": root,
                "original_request": (b.period, b.arrival_rank),
                "market": market,
                "deadline": passenger.deadline,
                "chosen": chosen,
                "stage": stage,
                "alpha": taste.alpha,
                "replacement_record": None,
                "fare": None,
            }
            if chosen is not None:
                request = Request(-1 - root, market, b.segment, b.taste_class, -1, -1)
                new = book(
                    self.world,
                    int(c.product_carrier[chosen]),
                    chosen,
                    float(prices[chosen]),
                    request,
                    c,
                )
                self._replacement_roots[new.record_id] = root
                self._passengers[root] = replace(passenger, active_record=new.record_id)
                record.update(replacement_record=new.record_id, fare=new.fare)
                purchases.append(new)
            self.rebooking_history.append(record)
        return tuple(purchases)

    def _stage_rescue_reenter(self, prices):
        if self.rescue_menu is None or self.rebooking is None:
            return self._stage_base_reenter(prices)
        (c, t) = (self.compiled, self.world.t)
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
            (private_prices, rescue) = self._rescue_prices(root, passenger, prices)
            (taste, gumbels) = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier] + ([None] if self.rebooking.interline else [])
            (chosen, stage) = (None, None)
            for carrier in stages:
                active = self._sellable_global().copy()
                for j in c.market_products[market]:
                    active[j] &= (
                        int(c.product_carrier[j]) == carrier
                        if carrier is not None
                        else int(c.product_carrier[j]) != b.carrier
                    )
                self._capture_stage_input(
                    root,
                    t,
                    market,
                    "own" if carrier is not None else "interline",
                    "rescue",
                    private_prices,
                    active,
                )
                utilities = systematic_utilities(
                    c, market, taste, private_prices, active
                )
                selected = sampled_choice(utilities, gumbels[list(columns)])
                self._capture_stage_result(root, t, columns, selected)
                if selected < len(columns) - 1:
                    chosen = columns[selected]
                    stage = "own" if carrier is not None else "interline"
                    break
            record = {
                "period": t,
                "original_record": root,
                "original_request": (b.period, b.arrival_rank),
                "market": market,
                "deadline": passenger.deadline,
                "chosen": chosen,
                "stage": stage,
                "alpha": taste.alpha,
                "replacement_record": None,
                "fare": None,
                "rescue_discount_used": False,
                **rescue,
            }
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
                record.update(
                    replacement_record=new.record_id,
                    fare=new.fare,
                    rescue_discount_used=bool(private_prices[chosen] < prices[chosen]),
                )
                purchases.append(new)
            self.rebooking_history.append(record)
        return tuple(purchases)

    def _reenter(self, prices):
        if self.seat_reserve is None or self.seat_reserve.share == 0:
            return self._stage_rescue_reenter(prices)
        self._sync_reserve()
        (c, t) = (self.compiled, self.world.t)
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
            (private_prices, rescue) = self._rescue_prices(root, passenger, prices)
            eligible = bool(rescue["rescue_eligible"])
            private_prices = private_prices.copy()
            for j in c.market_products[market]:
                if not any(self._held_on(j).values()):
                    private_prices[j] = prices[j]
            (taste, gumbels) = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier] + ([None] if self.rebooking.interline else [])
            (chosen, stage) = (None, None)
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
                    self._capture_stage_input(
                        root,
                        t,
                        market,
                        "own" if carrier is not None else "interline",
                        "reserve",
                        private_prices,
                        active,
                    )
                    utilities = systematic_utilities(
                        c, market, taste, private_prices, active
                    )
                    selected = sampled_choice(utilities, gumbels[list(columns)])
                    self._capture_stage_result(root, t, columns, selected)
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
                    held_seats_used={leg: n for (leg, n) in used.items() if n},
                )
                purchases.append(new)
            self.rebooking_history.append(record)
        return tuple(purchases)
