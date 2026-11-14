"""Exact ledger mutations and matched stochastic engine-path checks."""

from dataclasses import replace
import json
import numpy as np
import pytest

from airline_market.core.hashing import canonical_json
from airline_market.agents.random_coefficient import RandomCoefficientController
from airline_market.pipeline.scaling import scaling_network
from airline_market.sim.engine import MarketEngine
from airline_market.sim import ledger
from airline_market.sim.count_ledger import (
    CountedBookings,
    counted_ledger,
    counted_counts,
    counted_inventory,
)
from airline_market.sim.state import Booking
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.legacy_toy.model import Config
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec


def test_counted_list_mutations_match_fresh_reference_counts():
    c = compile_market(toy_hub())
    engine = MarketEngine(c)
    engine.reset(200000)
    state = engine.world.carriers[0]
    reference_counts, reference_inventory = ledger.counts, ledger.inventory
    products = c.products_of[0]
    a = Booking(1, 0, products[0], 100.0, 0, 0)
    b = Booking(2, 0, products[1], 120.0, 0, 1)
    state.bookings = CountedBookings([], c, 0)

    def check():
        assert np.array_equal(
            counted_counts(state, c, 0), reference_counts(state, c, 0)
        )
        assert np.array_equal(
            counted_inventory(state, c, 0), reference_inventory(state, c, 0)
        )

    state.bookings.append(a)
    state.bookings.extend([b, a])
    check()
    assert state.bookings.index(b) == 1
    state.bookings[::2] = [b, b]
    check()
    before = list(state.bookings)
    with pytest.raises(ValueError):
        state.bookings[::2] = [a]
    assert list(state.bookings) == before
    check()
    del state.bookings[1:2]
    state.bookings.insert(0, a)
    check()
    state.bookings.remove(b)
    state.bookings.pop()
    check()
    state.bookings += [a, b]
    state.bookings *= 2
    check()
    state.capacity[:] = 0
    check()
    state.bookings.clear()
    check()
    state.bookings = [a, b]
    check()
    copy = counted_counts(state, c, 0)
    copy[:] = 999
    check()


def trajectory(compiled, seed):
    engine = MarketEngine(
        compiled,
        {p.code: RandomCoefficientController() for p in compiled.spec.carriers},
    )
    engine.reset(seed)
    rows = []
    while not engine.terminal():
        engine.step()
        snapshot = engine.snapshot()
        state = engine.world
        rows.append(
            canonical_json(
                {
                    "carriers": state.carriers,
                    "requests": state.request_log,
                    "events": sorted(state.events_fired),
                    "prices": engine.trace(),
                    "background": state.background,
                    "observations": [
                        engine.observe(p.code, "linked") for p in compiled.spec.carriers
                    ],
                }
            )
        )
        engine.restore(snapshot)
    return rows, canonical_json(engine.metrics())


@pytest.mark.parametrize("n_carriers", [2, 3, 4])
def test_counted_engine_matches_reference_including_shocks_and_restore(n_carriers):
    s = scaling_network(
        {
            "n_spokes": 2,
            "n_carriers": n_carriers,
            "periods": 5,
            "controller_mode": "external_all",
            "requests_per_initial_seat": 3.0,
        }
    )
    event = CapacityEventSpec(
        trigger={"exo_transition": ["normal", "disrupted"]},
        effects=tuple((leg.id, -100) for leg in s.legs),
    )
    s = replace(
        s,
        capacity_process=CapacityProcessSpec(events=(event,)),
        exogenous=toy_hub(
            Config(horizon=5, shock_step=2, capacity_shock_probability=1)
        ).exogenous,
    )
    c = compile_market(s)
    baseline = [trajectory(c, seed) for seed in range(200000, 200006)]
    assert any(
        any(carrier["canceled"] for carrier in json.loads(rows[-1])["carriers"])
        for rows, _ in baseline
    )
    original = ledger.counts
    with counted_ledger():
        assert ledger.counts is counted_counts
        accelerated = [trajectory(c, seed) for seed in range(200000, 200006)]
    assert ledger.counts is original
    assert accelerated == baseline


def test_counted_scaling_child_uses_the_declared_implementation(tmp_path):
    """A subprocess must execute the counted worker and seal its correct label."""
    import json
    import os
    import subprocess
    import sys
    from airline_market.campaign.runner import THREAD_ENV

    config = {
        "schema": "airline-scaling-study/1",
        "id": "counted_worker_test",
        "revision": "counted-v1",
        "fixed": {"controller_mode": "external_all"},
        "cells": [{"n_spokes": 2, "n_carriers": 3, "periods": 3}],
        "n_cohorts": 2,
        "n_profile_cohorts": 1,
        "seed_start": 200000,
        "workers": 1,
        "limits": {"seconds": 30, "rss_gb": 2},
        "out": str(tmp_path / "run"),
        "log_path": str(tmp_path / "log.md"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "airline_market.experiments.scaling_counted",
            "--config",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, **THREAD_ENV},
    )
    result = json.loads((tmp_path / "run/results.json").read_text())["cells"][0]
    assert result["status"] == "completed"
    assert result["ledger_implementation"] == "airline_market.sim.count_ledger"
    assert "isolated counted" in result["engine"]
    assert sum(result["profile"]["exclusive_shares"].values()) == pytest.approx(1)
