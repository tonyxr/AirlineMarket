"""Native empty-period embedding, economic units, cent game and saved evidence."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.core.types import PriceAction
from airline_market.sim.engine import MarketEngine
from airline_market.spec import compile_market, load_spec

if os.environ.get("LOGIT_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.logit_bertrand_study",
        Path(os.environ["LOGIT_STAGE"]) / "logit_bertrand_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import logit_bertrand_study as study


def setup(label="symmetric"):
    path = Path(f"config/benchmarks/V4_logit_{label}_v1.json")
    c = compile_market(load_spec(path))
    return c, {
        "source_spec": str(path),
        "source_spec_sha256": file_hash(path),
        "compiled_spec_hash": c.spec_hash,
        "analytical_points": [
            [50.0, 50.0],
            [100.0, 110.0],
            [180.0, 70.0],
            [30.0, 250.0],
            [350.0, 350.0],
        ],
        "max_BR_iterations": 500,
        "sampling_prices": [100.0, 110.0],
        "validation_seed_start": 299900,
        "n_validation": 8,
    }


def test_known_equal_utilities_have_four_exact_kernel_categories():
    c, _ = setup()
    native = study.NativePayoff(c)
    assert native.probabilities([120.0, 120.0]) == pytest.approx(
        [0.85 / 3] * 3 + [0.15], abs=1e-12
    )
    assert study.closed_shares(native.q, [120.0, 120.0]) == pytest.approx(
        [1 / 3] * 3, abs=1e-12
    )


def test_empty_native_period_has_no_cash_stock_or_request_change():
    c, _ = setup()
    engine = MarketEngine(c)
    engine.reset(200011)
    before = [x.capacity.copy() for x in engine.world.carriers]
    step = engine.step({i: PriceAction({i: p}) for i, p in enumerate([50.0, 350.0])})
    assert not step.purchases and not step.resolutions and engine.world.t == 1
    assert engine.world.request_log == []
    assert all(
        x.cash == 0 and x.bookings == [] and np.array_equal(x.capacity, old)
        for x, old in zip(engine.world.carriers, before)
    )
    native = study.NativePayoff(c)
    altered = study.NativePayoff(c)
    altered.state = altered.kernel.encode(engine.world)
    assert native.probabilities([90.0, 110.0]) == pytest.approx(
        altered.probabilities([90.0, 110.0]), abs=1e-12
    )


@pytest.mark.parametrize("label", ["symmetric", "asymmetric"])
def test_native_global_cent_best_response_matches_exhaustive_closed_grid(label):
    c, _ = setup(label)
    native = study.NativePayoff(c)
    for player in [0, 1]:
        prices = [80.0, 190.0]
        br = native.best_response(prices, player)
        grid = np.arange(1000, 40001) / 100
        utility = native.q["values"][player] - native.q["alpha"] * grid
        other = 1 - player
        denominator = (
            1
            + np.exp(utility)
            + np.exp(native.q["values"][other] - native.q["alpha"] * prices[other])
        )
        values = (
            native.q["arrival"]
            * (grid - native.q["costs"][player])
            * np.exp(utility)
            / denominator
        )
        assert br["price"] == grid[np.argmax(values)]
        assert br["USD_profit"] == pytest.approx(float(values.max()), abs=1e-10)


@pytest.mark.parametrize("label", ["symmetric", "asymmetric"])
def test_all_registered_paths_close_to_continuous_FOC_and_all_points_preserved(label):
    c, p = setup(label)
    result = study.analytical(study.NativePayoff(c), p)
    assert result["analytical_point_pass"]
    assert len(result["points"]) == 5 and len(result["native_cent_BR_paths"]) == 8
    assert all(x["converged"] for x in result["native_cent_BR_paths"])
    if label == "symmetric":
        a, b = result["continuous_FOC_target"]["prices"]
        assert abs(a - b) < 1e-10


def test_contract_rejects_nonempty_opening_period():
    data = json.loads(Path("config/benchmarks/V4_logit_symmetric_v1.json").read_text())
    data["demand"]["arrival_probability"]["values"][0] = 0.1
    with pytest.raises(ValueError, match="one-active-period"):
        study.contract(compile_market(load_spec(data)))


def test_real_driver_has_completed_status_and_seals_every_native_row(tmp_path):
    _, p = setup()
    # A distinct physical fixture keeps test cohorts out of the production model.
    data = json.loads(Path(p["source_spec"]).read_text())
    for leg in data["legs"]:
        leg["capacity"]["support"] = [3]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    p.update(
        source_spec=str(fixture),
        source_spec_sha256=file_hash(fixture),
        compiled_spec_hash=compile_market(load_spec(fixture)).spec_hash,
    )
    result = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["status"] == "completed" and result["full_V4_pass"] is None
    assert result["sampler"]["cohorts"] == 8 and sum(result["sampler"]["counts"]) == 8
    rows = [
        json.loads(x)
        for x in (tmp_path / "native_cohorts.jsonl").read_text().splitlines()
    ]
    assert [r["seed"] for r in rows] == list(range(299900, 299908))
    for name, h in result["artifact_hashes"].items():
        assert file_hash(tmp_path / name) == h
    assert result["sampler"]["joint_Holm_decision"] is None


def test_model_hash_change_fails_before_model_construction(tmp_path):
    _, p = setup()
    p["source_spec_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_validation_split_rejects_final_seed_access(tmp_path):
    c, p = setup()
    p.update(validation_seed_start=299999, n_validation=2)
    with pytest.raises(ValueError, match="validation seed range"):
        study.sample(c, p, SimpleNamespace(directory=tmp_path))
    assert list(tmp_path.iterdir()) == []
