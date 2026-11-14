"""Isolated counted-ledger acceleration for F3; original source stays unchanged.

The reference count routine scans every booking once per product. This list
maintains the exact product counts and incidence usage on every mutation.
It changes a private representation, never the observation or pricing model.
"""

from contextlib import contextmanager
import sys
import numpy as np

from . import ledger


class CountedBookings(list):
    """A booking list with exact integer count/resource sufficient statistics."""

    def __init__(self, values, compiled, carrier):
        """Build counters from the current ledger, retaining immutable bookings."""
        super().__init__(values)
        self.compiled = compiled
        self.carrier = carrier
        self._product_index = {
            j: k for k, j in enumerate(compiled.products_of[carrier])
        }
        self.product_counts = np.zeros(len(self._product_index), dtype=np.int64)
        self.resource_used = np.zeros(len(compiled.legs_of[carrier]), dtype=np.int64)
        for booking in self:
            self._adjust(booking, 1)

    def _adjust(self, booking, sign):
        """Apply one immutable booking's incidence vector (Eq. 12)."""
        local = self._product_index.get(booking.product)
        if local is not None:
            self.product_counts[local] += sign
            self.resource_used += sign * self.compiled.A[self.carrier][:, local]

    def append(self, value):
        """Append a reservation and consume its complete itinerary."""
        super().append(value)
        self._adjust(value, 1)

    def extend(self, values):
        """Copy the iterable first so self-extension has normal list semantics."""
        values = list(values)
        super().extend(values)
        for value in values:
            self._adjust(value, 1)

    def insert(self, index, value):
        """Insert a booking while retaining exact stock counts."""
        super().insert(index, value)
        self._adjust(value, 1)

    def pop(self, index=-1):
        """Remove and return the booking selected by normal list indexing."""
        value = super().pop(index)
        self._adjust(value, -1)
        return value

    def remove(self, value):
        """Remove only the first equal booking, matching list behavior."""
        self.pop(super().index(value))

    def clear(self):
        """Clear both the active ledger and all sufficient statistics."""
        super().clear()
        self.product_counts.fill(0)
        self.resource_used.fill(0)

    def __setitem__(self, key, value):
        """Keep counters correct for individual and extended slice replacement."""
        old = self[key]
        values = list(value) if isinstance(key, slice) else value
        super().__setitem__(key, values)
        for booking in old if isinstance(key, slice) else [old]:
            self._adjust(booking, -1)
        for booking in values if isinstance(key, slice) else [values]:
            self._adjust(booking, 1)

    def __delitem__(self, key):
        """Delete the exact selected bookings after Python validates the index."""
        old = self[key]
        super().__delitem__(key)
        for booking in old if isinstance(key, slice) else [old]:
            self._adjust(booking, -1)

    def __iadd__(self, values):
        """Retain counted behavior for in-place list concatenation."""
        self.extend(values)
        return self

    def __imul__(self, multiple):
        """Match list repetition, including nonpositive multipliers."""
        super().__imul__(multiple)
        self.product_counts *= max(0, multiple)
        self.resource_used *= max(0, multiple)
        return self


def counted_bookings(state, compiled, carrier):
    """Attach counters after reset, clone, resolution or a replaced ledger list."""
    bookings = state.bookings
    if (
        not isinstance(bookings, CountedBookings)
        or bookings.compiled is not compiled
        or bookings.carrier != carrier
    ):
        bookings = CountedBookings(bookings, compiled, carrier)
        state.bookings = bookings
    return bookings


def counted_counts(state, compiled, carrier):
    """Return a fresh exact vector, preserving reference API mutation semantics."""
    return counted_bookings(state, compiled, carrier).product_counts.copy()


def counted_inventory(state, compiled, carrier):
    """Use current physical capacity minus maintained exact integer usage."""
    return state.capacity - counted_bookings(state, compiled, carrier).resource_used


@contextmanager
def counted_ledger():
    """Install equivalent primitives only inside the current isolated process.

    Imported aliases in maintained modules are restored on exit, including
    aliases imported while the context was active. Legacy/vendor modules and
    source files are never edited. Run this only in a dedicated study worker.
    """
    original_counts, original_inventory = ledger.counts, ledger.inventory
    if original_counts is counted_counts:
        yield
        return
    replacements = {
        original_counts: counted_counts,
        original_inventory: counted_inventory,
    }

    def modules():
        """Select only maintained modules loaded in this Python process."""
        return [
            module
            for name, module in list(sys.modules.items())
            if module is not None
            and name.startswith("airline_market.")
            and name != __name__
            and not name.startswith(
                ("airline_market.vendor.", "airline_market.legacy_toy.")
            )
        ]

    for module in modules():
        for name, value in list(vars(module).items()):
            for original, replacement in replacements.items():
                if value is original:
                    setattr(module, name, replacement)
    try:
        yield
    finally:
        for module in modules():
            for name, value in list(vars(module).items()):
                for original, replacement in replacements.items():
                    if value is replacement:
                        setattr(module, name, original)
