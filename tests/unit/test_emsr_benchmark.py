"""Independent normal reference, native threshold capture and absence of simulation."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.agents.emsr import EMSRbController
from airline_market.campaign.artifacts import file_hash
from airline_market.core.hashing import schema_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("EMSR_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.emsr_benchmark_study",
        Path(os.environ["EMSR_STAGE"]) / "emsr_benchmark_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import emsr_benchmark_study as study


def setup(cap=164):
    cfg = json.loads(Path("config/campaigns/V4_EMSR_fourclass.json").read_text())
    cell = next(x for x in cfg["cells"] if x["capacity"] == cap)
    p = {**cfg["fixed"], **cell}
    return compile_market(load_spec(p["source_spec"])), p


def test_independent_reference_uses_no_native_helper(monkeypatch):
    monkeypatch.setattr(
        study,
        "emsr_b_protection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("native helper")),
    )
    levels, rows = study.independent_reference(
        [100.0, 50.0, 40.0, 20.0], [10.0] * 4, [2.0, 3.0, 4.0, 5.0], 100
    )
    assert levels[1] == 10.0 and rows[0]["normal_quantile"] == 0.0


@pytest.mark.parametrize("cap", [164, 124])
def test_actual_native_controller_thresholds_match_independent_normal_formula(cap):
    c, p = setup(cap)
    reference, rows = study.independent_reference(
        p["fares"], p["means"], p["stds"], cap
    )
    assert [round(x["raw_protection_seats"], 1) for x in rows] == [16.7, 51.5, 131.4]
    ctrl = study.build_controller(c, p)
    for stock in [0, 16, 17, 51, 52, 124, cap]:
        obs = study.permitted_observation(cap, stock)
        before = schema_hash(obs)
        action, native = study.capture_actual_thresholds(ctrl, obs)
        assert native == pytest.approx(reference, abs=1e-8)
        assert schema_hash(obs) == before
        eligible = [k for k, x in enumerate(reference) if stock > x]
        assert action.prices[0] == p["fares"][max(eligible) if eligible else 0]
    assert reference[-1] == min(float(cap), rows[-1]["raw_protection_seats"])


def test_existing_profile_is_chained_and_restored():
    c, p = setup()
    ctrl = study.build_controller(c, p)
    events = []
    old = sys.getprofile()

    def previous(frame, event, arg):
        if frame.f_code is EMSRbController.act.__code__:
            events.append(event)

    try:
        sys.setprofile(previous)
        study.capture_actual_thresholds(ctrl, study.permitted_observation(164, 80))
        assert sys.getprofile() is previous and "return" in events
    finally:
        sys.setprofile(old)


def test_profile_restored_when_controller_raises():
    class Raising(EMSRbController):
        def act(self, obs):
            raise RuntimeError("unit exception")

    c, p = setup()
    ctrl = study.build_controller(c, p)
    ctrl.__class__ = Raising
    previous = sys.getprofile()
    with pytest.raises(RuntimeError, match="unit exception"):
        study.capture_actual_thresholds(ctrl, study.permitted_observation(164, 80))
    assert sys.getprofile() is previous


def test_own_observation_ledger_is_consistent_and_has_no_hidden_inputs():
    obs = study.permitted_observation(164, 50)
    assert len(obs.own_bookings) == 114 and obs.own_cash == 114 * 527.0
    assert obs.privileged is None and obs.quotes == ()
    assert [b.period for b in obs.own_bookings] == list(range(114))
    assert len(obs.own_bookings) + obs.own_inventory[0] == obs.own_capacity[0]
    with pytest.raises(ValueError, match="integer inventory"):
        study.permitted_observation(164, 165)


def test_invalid_four_class_reference_is_rejected_without_epsilon_repair():
    with pytest.raises(ValueError, match="four strictly"):
        study.independent_reference(
            [100.0, 100.0, 50.0, 20.0], [10.0] * 4, [2.0] * 4, 100
        )


def test_driver_seals_all_inventory_observations_without_engine_or_tape(
    tmp_path, monkeypatch
):
    from airline_market.sim.engine import MarketEngine
    from airline_market.sim.tape import ExogenousTape

    def forbidden(*args, **kwargs):
        raise AssertionError("physical simulator/tape called")

    monkeypatch.setattr(MarketEngine, "reset", forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    _, p = setup(124)
    # A distinct small physical fixture checks the driver without repeating a production sweep.
    data = json.loads(Path(p["source_spec"]).read_text())
    data["legs"][0]["capacity"]["support"] = [7]
    data["horizon"]["n_periods"] = 9
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    p.update(
        capacity=7,
        source_spec=str(fixture),
        source_spec_sha256=file_hash(fixture),
        compiled_spec_hash=compile_market(load_spec(fixture)).spec_hash,
    )
    result = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["status"] == "completed" and result["EMSR_fourclass_case_pass"]
    assert result["controller_observations"] == 8 and result["simulation_cohorts"] == 0
    assert result["full_V4_pass"] is None and result["action_mismatches"] == 0
    for name, h in result["artifact_hashes"].items():
        assert file_hash(tmp_path / name) == h


def test_input_hash_guard_before_any_output(tmp_path):
    _, p = setup()
    p["native_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen EMSR input"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert not list(tmp_path.iterdir())
