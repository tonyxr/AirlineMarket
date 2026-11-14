"""Sparse access must preserve exact observable paths, not just mean outcomes."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from airline_market.agents.price_rule import FeatureEvaluator
from airline_market.agents.random_coefficient import RandomCoefficientController
from airline_market.campaign.runner import THREAD_ENV
from airline_market.core.hashing import canonical_json
from airline_market.pipeline.scaling import scaling_network
from airline_market.sim.engine import MarketEngine
from airline_market.sim.sparse_access import SparseFeatureEvaluator, sparse_access
from airline_market.spec import compile_market
from airline_market.spec.model import (
    FeatureSpec,
    CapacityEventSpec,
    CapacityProcessSpec,
)
from airline_market.spec.library_builders import toy_hub
from airline_market.legacy_toy.model import Config


def path(compiled, seed):
    """Capture prices, offers, accounting, redacted observations and restored state."""
    engine = MarketEngine(
        compiled,
        {p.code: RandomCoefficientController() for p in compiled.spec.carriers},
    )
    engine.reset(seed)
    rows = []
    while not engine.terminal():
        engine.step()
        snapshot = engine.snapshot()
        rows.append(
            canonical_json(
                {
                    "carriers": engine.world.carriers,
                    "requests": engine.world.request_log,
                    "events": sorted(engine.world.events_fired),
                    "background": engine.world.background,
                    "quotes": engine.world.quote_history.records,
                    "initial_quotes": engine.world.quote_history.initial,
                    "observations": [
                        engine.observe(p.code, "linked") for p in compiled.spec.carriers
                    ],
                    "trace": engine.trace(),
                }
            )
        )
        engine.restore(snapshot)
    return rows, canonical_json(engine.metrics())


@pytest.mark.parametrize("carriers", [2, 3, 4])
def test_sparse_exact_paths_with_canceling_capacity_shock(carriers):
    spec = scaling_network(
        {
            "n_spokes": 3,
            "n_carriers": carriers,
            "periods": 5,
            "controller_mode": "external_all",
            "requests_per_initial_seat": 4.0,
        }
    )
    spec = replace(
        spec,
        exogenous=toy_hub(
            Config(horizon=5, shock_step=2, capacity_shock_probability=1)
        ).exogenous,
        capacity_process=CapacityProcessSpec(
            events=(
                CapacityEventSpec(
                    trigger={"exo_transition": ["normal", "disrupted"]},
                    effects=tuple((leg.id, -100) for leg in spec.legs),
                ),
            )
        ),
    )
    compiled = compile_market(spec)
    expected = [path(compiled, seed) for seed in range(200020, 200024)]
    assert any(
            any(c["canceled"] for c in json.loads(rows[-1])["carriers"])
        for rows, _ in expected
    )
    with sparse_access():
        actual = [path(compiled, seed) for seed in range(200020, 200024)]
    assert actual == expected


def test_sparse_feature_values_are_exact_for_all_own_products():
    compiled = compile_market(scaling_network({"n_spokes": 4, "periods": 3}))
    engine = MarketEngine(compiled)
    engine.reset(200025)
    for carrier in range(len(compiled.spec.carriers)):
        obs = engine.observe(compiled.spec.carriers[carrier].code, "linked")
        reference = FeatureEvaluator(compiled, carrier)
        sparse = SparseFeatureEvaluator(compiled, carrier)
        features = [
            FeatureSpec(name=name)
            for name in [
                "intercept",
                "product_indicator",
                "connection_indicator",
                "scarcity_sum",
                "scarcity_max",
                "time_to_departure",
                "booking_pace",
                "min_rival_gap",
                "rival_availability",
                "public_disruption",
                "linked_scarcity_signal",
            ]
        ]
        for j in compiled.products_of[carrier]:
            assert [sparse(f, j, obs) for f in features] == [
                reference(f, j, obs) for f in features
            ]


def test_sparse_child_receipt_and_profile_label(tmp_path):
    config = {
        "schema": "airline-scaling-study/1",
        "id": "sparse_worker_test",
        "revision": "sparse-v1",
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
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "airline_market.experiments.scaling_sparse",
            "--config",
            str(cfg),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, **THREAD_ENV},
    )
    result = json.loads((tmp_path / "run/results.json").read_text())["cells"][0]
    assert result["status"] == "completed"
    assert result["sparse_implementation"] == "airline_market.sim.sparse_access"
    assert sum(result["profile"]["exclusive_shares"].values()) == pytest.approx(1)
    assert all(x >= 0 for x in result["profile"]["exclusive_shares"].values())
