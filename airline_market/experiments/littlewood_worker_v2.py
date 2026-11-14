"""Correct the CellContext counter integration without altering frozen RM math."""

from types import SimpleNamespace

from airline_market.experiments.littlewood_study import run as frozen_run


class CounterPublisher:
    """Publish to the actual worker's documented counters callback."""

    def __init__(self, context):
        self.current = {}
        context.counters = lambda: dict(self.current)

    def update(self, **values):
        """Set current counters without changing simulation or RNG state."""
        self.current.update(values)


def run(cell, context):
    """Use frozen v1 science with an explicitly compatible worker-counter view."""
    proxy = SimpleNamespace(
        directory=context.directory, counter=CounterPublisher(context)
    )
    return frozen_run(cell, proxy)
