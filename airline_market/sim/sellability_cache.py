"""Exact cached availability for immutable binary itinerary incidence.

For nonnegative stock and binary incidence, a product is sellable exactly
when it uses none of the legs with stock below one. Full positive stock
levels still enter prices and quotes; this cache replaces only feasibility.
"""

from contextlib import contextmanager
from collections import OrderedDict
import sys
import numpy as np

from . import ledger


class BinarySellabilityCache:
    """Cache stock-exhaustion masks without changing the primitive's return ownership."""

    def __init__(self, max_entries=8192):
        """Keep strong immutable matrix references and a bounded exact-result cache."""
        self.max_entries = max_entries
        self.matrices = {}
        self.cache = OrderedDict()
        self.calls = self.hits = self.misses = 0

    def __call__(self, inventory, matrix):
        """Return the same fresh boolean vector as dense binary comparisons."""
        identity = id(matrix)
        if identity not in self.matrices:
            if matrix.flags.writeable or np.any((matrix != 0) & (matrix != 1)):
                raise ValueError("cached sellability requires immutable binary incidence")
            self.matrices[identity] = matrix
        values = np.asarray(inventory)
        self.calls += 1
        if np.any(values < 0):
            return np.zeros(matrix.shape[1], dtype=bool)
        blocked = values < 1
        key = identity, np.packbits(blocked).tobytes()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        self.misses += 1
        result = ~np.any(matrix[blocked, :], axis=0)
        result.setflags(write=False)
        self.cache[key] = result
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
        return result.copy()

    def info(self):
        """Report physical feasibility work units, separate from solver nodes."""
        return {"calls": self.calls, "hits": self.hits, "misses": self.misses,
                "entries": len(self.cache), "immutable_matrices": len(self.matrices)}


@contextmanager
def cached_sellability():
    """Patch process-local aliases only; restore them even after an exception."""
    original = ledger.sellable
    cache = BinarySellabilityCache()

    def modules():
        """Include aliases imported in this dedicated worker, excluding frozen sources."""
        return [m for name, m in list(sys.modules.items()) if m is not None
                and name.startswith("airline_market.") and name != __name__
                and not name.startswith(("airline_market.legacy_toy.", "airline_market.vendor."))]

    for module in modules():
        for name, value in list(vars(module).items()):
            if value is original:
                setattr(module, name, cache)
    try:
        yield cache
    finally:
        for module in modules():
            for name, value in list(vars(module).items()):
                if value is cache:
                    setattr(module, name, original)
