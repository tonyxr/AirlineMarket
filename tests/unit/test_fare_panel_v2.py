"""B2/H2.1: new native UTC clock binding, retain v1 limitations and parent bytes."""

import importlib.util
import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest

from airline_market.policy.dated_reserve import UtcBookingClock
from tests.unit import test_dated_shocks as dated
from tests.unit import test_fare_panel as parent

if os.environ.get("B2_PANEL_V2_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_fare_panel_v2", os.environ["B2_PANEL_V2_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.eval import fare_panel_v2 as module


def test_actual_dated_engine_matching_clock_projects_and_measures_once(tmp_path):
    e = dated.fixture(policy=True)
    parent.finish(e)
    c = e.shock_clock
    d = parent.definition(e)
    p = parent.pairing(e, c, d)
    directory = tmp_path / "dated"
    directory.mkdir()
    artifact = module.capture_native_posted_panel(
        e,
        c,
        d,
        p,
        run_id="unit-baseline",
        channels=None,
        destination=directory / "panel.json",
    )
    src = parent.bind_result(directory, artifact)
    assert parent.values(
        module.read_saved_fare_panel(
            src, e.compiled, c, d, p, run_id="unit-baseline", channels=None
        )
    ) == {"mean_fare": 250, "p90_fare": 300}


def test_v2_refuses_wrong_shock_clock_before_any_source_is_written(tmp_path):
    e = dated.fixture()
    parent.finish(e)
    times = list(e.shock_clock.boundaries)
    times[1] += timedelta(minutes=1)
    c = UtcBookingClock(tuple(times))
    d = parent.definition(e)
    p = parent.pairing(e, c, d)
    with pytest.raises(ValueError, match="native public shock_clock"):
        module.capture_native_posted_panel(
            e,
            c,
            d,
            p,
            run_id="unit-baseline",
            channels=None,
            destination=tmp_path / "wrong.json",
        )
    assert not (tmp_path / "wrong.json").exists()


def test_v2_checks_booking_clock_as_well_as_shock_clock(tmp_path):
    e = parent.finish(parent.engine())
    c = parent.clock()
    times = list(c.boundaries)
    times[1] += timedelta(minutes=1)
    view = SimpleNamespace(
        compiled=e.compiled,
        tape=e.tape,
        trace=e.trace,
        terminal=e.terminal,
        shock_clock=c,
        booking_clock=UtcBookingClock(tuple(times)),
    )
    d = parent.definition(e)
    p = parent.pairing(e, c, d)
    with pytest.raises(ValueError, match="native public booking_clock"):
        module.capture_native_posted_panel(
            view,
            c,
            d,
            p,
            run_id="unit-baseline",
            channels=None,
            destination=tmp_path / "wrong-policy.json",
        )
    assert not (tmp_path / "wrong-policy.json").exists()


def test_undated_native_base_keeps_explicit_caller_clock_scope(tmp_path):
    e = parent.finish(parent.engine())
    c = parent.clock()
    d = parent.definition(e)
    p = parent.pairing(e, c, d)
    directory = tmp_path / "base"
    directory.mkdir()
    artifact = module.capture_native_posted_panel(
        e,
        c,
        d,
        p,
        run_id="unit-baseline",
        channels=None,
        destination=directory / "panel.json",
    )
    out = module.read_saved_fare_panel(
        parent.bind_result(directory, artifact),
        e.compiled,
        c,
        d,
        p,
        run_id="unit-baseline",
        channels=None,
    )
    assert parent.values(out) == {"mean_fare": 250, "p90_fare": 300}
    assert not out.record()["provenance"]["original_time_provenance_authenticated"]
