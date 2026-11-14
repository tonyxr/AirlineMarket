"""MR1 fixed-other-fare exact queries, parent immutability and full driver contract."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.core.rng import StreamRegistry
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, load_spec

if os.environ.get("PRICE_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.price_monotonicity_study",
        Path(os.environ["PRICE_STAGE"]) / "price_monotonicity_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import price_monotonicity_study as study


def fixture():
    c = compile_market(load_spec("config/benchmarks/V4_logit_symmetric_v1.json"))
    engine = MarketEngine(c)
    engine.reset(200009)
    engine.step(study.actions(c, [100.0, 110.0]))
    return c, engine, ExactKernel(c)


def test_native_shares_and_monotonicity_with_other_posted_fare_fixed():
    c, e, k = fixture()
    state = k.encode(e.world)
    base, mass, _ = study.expected_new_bookings(k, state, [120.0, 120.0], 1e-12)
    raised, _, _ = study.expected_new_bookings(k, state, [130.0, 120.0], 1e-12)
    assert base == pytest.approx([0.85 / 3] * 2, abs=1e-12) and abs(mass - 1) < 1e-12
    row = study.compare_change(c, base, raised, 0, 1e-12)
    assert (
        row["pass"]
        and row["own_product_delta"] < 0
        and row["rival_aggregate_delta"] > 0
    )


def test_parent_state_tape_controller_memory_and_RNG_unchanged():
    _c, e, k = fixture()
    before = study.parent_fingerprint(e, k)
    for prices in [[100.0, 110.0], [400.0, 110.0], [100.0, 400.0]]:
        study.expected_new_bookings(k, k.encode(e.world), prices, 1e-12)
        assert study.parent_fingerprint(e, k) == before


def test_negative_and_numerical_zero_rows_are_not_filtered():
    c, _, _ = fixture()
    base = np.array([0.2, 0.3])
    assert not study.compare_change(c, base, np.array([0.25, 0.29]), 0, 1e-12)["pass"]
    row = study.compare_change(c, base, base, 0, 1e-12)
    assert (
        row["pass"]
        and row["own_product_delta"] == 0
        and row["rival_aggregate_delta"] == 0
    )


def test_raw_mass_failure_is_not_renormalized(monkeypatch):
    _, e, k = fixture()
    monkeypatch.setattr(
        k,
        "period_outcomes",
        lambda *args: [(0.9, SimpleNamespace(requests=(), choices=()))],
    )
    with pytest.raises(ValueError, match="no renormalization"):
        study.expected_new_bookings(k, k.encode(e.world), [100.0, 100.0], 1e-12)


def test_only_one_product_is_raised_and_all_actions_keep_the_other_fares():
    c, _, _ = fixture()
    a = study.actions(c, [50.0, 100.0])
    assert a[0].prices == {0: 50.0} and a[1].prices == {1: 100.0}
    rng = StreamRegistry(773, "unit/cent-vector").generator()
    for _ in range(30):
        v = study.cent_vector([(1000, 1003), (1200, 1202)], rng)
        assert 10 <= v[0] <= 10.02 and 12 <= v[1] <= 12.01


def test_full_driver_on_distinct_small_unit_fixture_and_sealed_rows(tmp_path):
    source = Path("config/benchmarks/V4_logit_symmetric_v1.json")
    data = json.loads(source.read_text())
    for leg in data["legs"]:
        leg["capacity"]["support"] = [3]
    spec = tmp_path / "unit_model.json"
    spec.write_text(json.dumps(data))
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    c = compile_market(load_spec(spec))
    p = {
        "instance": "distinct_unit_fixture",
        "source_spec": str(spec),
        "source_spec_sha256": file_hash(spec),
        "compiled_spec_hash": c.spec_hash,
        "design_registration": str(registration),
        "design_registration_sha256": file_hash(registration),
        "n_states": 6,
        "seed_start": 299900,
        "tolerance": 1e-12,
    }
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        r["status"] == "completed"
        and r["MR1_saved_instance_pass"]
        and r["reachable_states"] == 6
    )
    assert r["comparisons"] == 12 and r["complete_physical_horizon_cohorts"] == 0
    rows = [
        json.loads(x)
        for x in (tmp_path / "state_comparisons.jsonl").read_text().splitlines()
    ]
    assert [x["seed"] for x in rows] == list(range(299900, 299906))
    for row in rows:
        assert row["parent_unchanged"]
        for change in row["changes"]:
            j = change["changed_product"]
            assert change["raised_prices"][j] > row["base_prices"][j]
            assert all(
                x == y
                for i, (x, y) in enumerate(
                    zip(row["base_prices"], change["raised_prices"])
                )
                if i != j
            )
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )
    assert r["full_V3_pass"] is None


def test_source_guard_and_validation_seed_guard_before_outputs(tmp_path):
    p = {"seed_start": 299999, "n_states": 2}
    with pytest.raises(ValueError, match="validation"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    p.update(
        seed_start=299990,
        source_spec="config/benchmarks/V4_logit_symmetric_v1.json",
        source_spec_sha256="0" * 64,
        design_registration="config/benchmarks/V4_logit_symmetric_v1.json",
        design_registration_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="frozen"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert list(tmp_path.iterdir()) == []
