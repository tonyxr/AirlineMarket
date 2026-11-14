"""B2/H2.1: bind panel capture to every exposed native public UTC clock.

Retain v1 bytes/results. v1 accepts a caller clock without checking native dated
policy clocks; this prospective wrapper refuses disagreement before export.
"""

from airline_market.eval.fare_panel import (
    FarePanelDefinition,
    FarePanelMetrics,
    SavedFarePanelSource,
    panel_contract,
    read_saved_fare_panel,
)
from airline_market.eval.fare_panel import (
    capture_native_posted_panel as _capture,
)

__all__ = (
    "FarePanelDefinition",
    "FarePanelMetrics",
    "SavedFarePanelSource",
    "capture_native_posted_panel",
    "panel_contract",
    "read_saved_fare_panel",
)


def capture_native_posted_panel(
    engine, clock, definition, contract, *, run_id, channels, destination
):
    """Check public clocks only, then delegate the ONE native trace projection.

    An undated base engine still needs the caller's explicit clock declaration;
    this does not authenticate public dates/visibility or economic M/F/S channels.
    """
    for name in ("shock_clock", "booking_clock"):
        native = getattr(engine, name, None)
        if native is not None and native != clock:
            raise ValueError("fare panel clock differs from native public " + name)
    return _capture(
        engine,
        clock,
        definition,
        contract,
        run_id=run_id,
        channels=channels,
        destination=destination,
    )
