"""Checks for concrete accounting errors, not policy/performance gates."""

import math
from .ledger import inventory


def check_world(world, compiled):
    """
    Verify seat conservation, cash ledgers, unique records and quote coverage (Eqs.
    3–16).
    """
    ids = []
    for i, state in enumerate(world.carriers):
        assert min(inventory(state, compiled, i), default=0) >= 0, (
            "negative leg inventory"
        )
        assert (
            abs(state.cash - (state.revenue - state.refunds - state.penalties)) <= 1e-9
        ), "cash ledger mismatch"
        ids.extend(b.record_id for b in state.bookings)
    assert len(ids) == len(set(ids)), "duplicate active booking IDs"
    assert all(b.stock >= 0 for b in world.background), "negative reference-route stock"
    completed = sum(t < world.t for t in world.quote_history.records)
    assert completed == world.t, "posted quote history is incomplete"
