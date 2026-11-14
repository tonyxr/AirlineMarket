"""Purchase-contingent core resources for the existing OUTSIDE alternative.

One potential passenger still makes one native random-utility choice. A private,
pre-drawn outside subtype can require core seats; an infeasible subtype removes
that alternative before utility maximization. Its purchase atomically attaches
zero-receipt core components to the single outside ticket. Existing engines,
solvers and learner source files are unchanged.
"""

import copy
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

from . import engine as native
from . import ledger
from .arrivals import period_requests
from .shared_background import SharedLoadEngine


@dataclass(frozen=True)
class PurchaseFootprint:
    """An identified dated resource footprint, or an explicitly unmapped subtype."""

    legs: tuple[str, ...]
    mapped: bool = True


@dataclass(frozen=True)
class PurchaseSnapshot:
    """Native/external-load state plus single-ticket component links and counters."""

    native: object
    links: tuple
    counters: tuple


class CoupledOutsideEngine(SharedLoadEngine):
    """Keep excluded in-pool resource use attached to its native ticket purchase."""

    def __init__(self, compiled, events=(), controllers=None, *, footprint_tape=None,
                 outside_carrier="OUTSIDE", fraction=1.0, check_invariants=True):
        """Bind a policy-independent private tape keyed by period/potential-request row."""
        super().__init__(compiled, events, controllers, fraction=fraction,
                         check_invariants=check_invariants)
        self.footprint_tape = MappingProxyType(dict(footprint_tape or {}))
        self.outside_carrier = outside_carrier
        self.outside_index = compiled.carrier_index[outside_carrier]
        self.outside_products = {}
        for j in compiled.products_of[self.outside_index]:
            m = int(compiled.product_market[j])
            if m in self.outside_products:
                raise ValueError("purchase coupling requires one aggregate outside offer per market")
            self.outside_products[m] = j
        for (t, row), mark in self.footprint_tape.items():
            if (not 0 <= t < compiled.spec.horizon.n_periods or row < 0
                    or any(l not in self.load_products for l in mark.legs)):
                raise ValueError("invalid purchase footprint tape")
        self.group_base = max((e.id for e in self.events), default=-1) + 1
        self.purchase_links, self.purchase_counts, self._pending = {}, Counter(), None

    def reset(self, seed=None, *, tape=None):
        """Reset purchase links without drawing or exposing any future subtype."""
        self.purchase_links, self.purchase_counts, self._pending = {}, Counter(), None
        super().reset(seed, tape=tape)

    def _feasible(self, mark):
        """Test the complete physical footprint before a passenger chooses."""
        if not mark.mapped:
            return False
        c = self.compiled
        for leg, amount in Counter(mark.legs).items():
            global_leg = c.leg_index[leg]
            carrier = int(c.leg_carrier[global_leg])
            local = c.legs_of[carrier].index(global_leg)
            if ledger.inventory(self.world.carriers[carrier], c, carrier)[local] < amount:
                return False
        return True

    def _annotated_requests(self, requests):
        """Select only this request's private subtype; no new passenger is generated."""
        try:
            for request in requests:
                mark = self.footprint_tape.get((self.world.t, request.gumbel_row))
                target = self.outside_products.get(request.market)
                self._pending = (request, mark, target)
                if mark is not None:
                    if target is None:
                        raise ValueError("purchase footprint has no outside offer in its market")
                    self.purchase_counts["identified_path_opportunities"] += 1
                    if not mark.mapped:
                        self.purchase_counts["unmapped_path_opportunities"] += 1
                    elif not self._feasible(mark):
                        self.purchase_counts["capacity_blocked_opportunities"] += 1
                yield request
        finally:
            self._pending = None

    def _sellable_global(self):
        """Apply subtype capacity before native utility, choice and welfare calculations."""
        active = super()._sellable_global()
        if self._pending is not None:
            _, mark, target = self._pending
            if mark is not None and not self._feasible(mark):
                active[target] = False
        return active

    def _book_choice(self, original, world, carrier, product, fare, request, compiled):
        """Book one native ticket and all attached core seats as one transaction."""
        mark = target = None
        if self._pending is not None:
            _, mark, target = self._pending
        if mark is None or product != target:
            return original(world, carrier, product, fare, request, compiled)
        if not self._feasible(mark):
            raise ValueError("forced outside choice exceeds its complete resource footprint")
        ticket = original(world, carrier, product, fare, request, compiled)
        components = []
        placeholder = SimpleNamespace(rank=-1, segment=-1, taste_class=-1)
        for leg in mark.legs:
            j = self.load_products[leg]
            i = int(compiled.product_carrier[j])
            b = ledger.book(world, i, j, 0.0, placeholder, compiled)
            components.append((i, b.record_id))
            global_leg = compiled.leg_index[leg]
            local = compiled.legs_of[i].index(global_leg)
            if ledger.inventory(world.carriers[i], compiled, i)[local] == 0:
                world.sellout_period.setdefault(global_leg, world.t)
        identifier = self.group_base + ticket.record_id
        self.groups[identifier] = tuple(components)
        self.purchase_links[identifier] = (carrier, ticket.record_id)
        self.load_counts["attempted_groups"] += 1
        self.load_counts["accepted_groups"] += 1
        self.load_counts["accepted_components"] += len(components)
        self.purchase_counts["coupled_ticket_purchases"] += 1
        self.purchase_counts["coupled_core_components"] += len(components)
        return ticket

    def reconcile_load_groups(self):
        """Cancel/refund the outside ticket when any attached component is lost."""
        live = {(i, b.record_id): b for i, s in enumerate(self.world.carriers) for b in s.bookings}
        for identifier, ticket_key in self.purchase_links.items():
            if identifier in self.canceled_groups or ticket_key in live:
                continue
            self.canceled_groups.add(identifier)
            for i, record in self.groups[identifier]:
                if (i, record) in live:
                    state = self.world.carriers[i]
                    state.bookings = [b for b in state.bookings if b.record_id != record]
                    state.canceled += 1
        super().reconcile_load_groups()
        for identifier, (i, record) in self.purchase_links.items():
            if identifier not in self.canceled_groups:
                continue
            state = self.world.carriers[i]
            ticket = next((b for b in state.bookings if b.record_id == record), None)
            if ticket is not None:
                state.bookings = [b for b in state.bookings if b.record_id != record]
                state.cash -= ticket.fare
                state.refunds += ticket.fare
                state.canceled += 1
        self.purchase_counts["canceled_coupled_tickets"] = sum(
            identifier in self.canceled_groups for identifier in self.purchase_links)

    @contextmanager
    def _native_transaction(self):
        """Intercept only this world's bookings/events in the dedicated native worker."""
        previous_book, previous_events = native.book, native.apply_events

        def record(world, carrier, product, fare, request, compiled):
            """Route only this engine's native ticket through the attachment transaction."""
            if world is self.world:
                return self._book_choice(previous_book, world, carrier, product,
                                         fare, request, compiled)
            return previous_book(world, carrier, product, fare, request, compiled)

        def resolve(world, compiled, before, after, t, tape):
            """Settle attachments before native cash-change callbacks are delivered."""
            result = previous_events(world, compiled, before, after, t, tape)
            if world is self.world:
                self.reconcile_load_groups()
            return result

        native.book, native.apply_events = record, resolve
        try:
            yield
        finally:
            native.book, native.apply_events = previous_book, previous_events

    def step(self, actions=None, *, _requests=None, _choices=None):
        """Keep native request count, random utilities, receipts and callback timing."""
        requests = (period_requests(self.compiled, self.tape, self.world.t, self.world.exo)
                    if _requests is None else _requests)
        annotated = self._annotated_requests(requests)
        try:
            with self._native_transaction():
                return super().step(actions, _requests=annotated, _choices=_choices)
        finally:
            annotated.close()
            self._pending = None

    def metrics(self):
        """Report one ticket per choice, with physical attachments separated from revenue."""
        result = super().metrics()
        result["outside_resource_coupling"] = dict(self.purchase_counts)
        return result

    def snapshot(self):
        """Retain receipt/component linkage as well as parent replay state."""
        return PurchaseSnapshot(super().snapshot(), tuple(self.purchase_links.items()),
                                tuple(self.purchase_counts.items()))

    def restore(self, snapshot):
        """Restore between-request state without reselecting any outside subtype."""
        super().restore(snapshot.native)
        self.purchase_links = dict(snapshot.links)
        self.purchase_counts = Counter(dict(snapshot.counters))
        self._pending = None

    def branch(self):
        """Branch with the same private subtype tape and independent mutable ledgers."""
        other = CoupledOutsideEngine(self.compiled, self.events,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            footprint_tape=self.footprint_tape, outside_carrier=self.outside_carrier,
            fraction=self.fraction, check_invariants=self.check_invariants)
        other.restore(self.snapshot())
        return other
