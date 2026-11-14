"""S1/R4: constructor randomness is not silently labelled zero RNG activity."""

import importlib.util
import json
import os

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.worker import CellContext

if os.environ.get("S1_METRICS_RNG_FIX_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_rng_fix", os.environ["S1_METRICS_RNG_FIX_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    from airline_market.experiments import screen_metrics_study_v2 as module


def test_corrected_wrapper_preserves_metric_values_and_qualifies_rng(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(module, "verify_inputs", lambda p: calls.append("verify"))
    monkeypatch.setattr(
        module,
        "analyze_settled_setting",
        lambda *args: {
            "pairwise_information": {"unchanged": True},
            "policy_divergence": {"first_period": None},
        },
    )
    p = {
        "setting_id": "new_rng_fixture",
        "source_root": "local",
        "policy_hash_scope": "post-completion snapshot",
    }
    context = CellContext(tmp_path)
    r = module.run(
        {"parameters": p, "limits": {"rss_gb": 0.5, "reservation_gb": 0.5}}, context
    )
    assert calls == ["verify", "verify"]
    m = json.loads((tmp_path / r["metrics_path"]).read_text())
    assert (
        m["pairwise_information"] == {"unchanged": True}
        and m["policy_divergence"]["first_period"] is None
    )
    for value in [r, m]:
        assert (
            value["new_rng_calls"] is None
            and value["reference_constructor_generates_tapes"] is True
        )
        assert (
            value["reference_construction_seed"] == 0
            and value["traversal_sampling_calls"]
            == value["sampled_physical_cohorts"]
            == 0
        )
    assert file_hash(tmp_path / r["metrics_path"]) == r["metrics_sha256"]


def test_actual_history_dp_construction_generates_reference_tapes(monkeypatch):
    from airline_market.pipeline.screen_generators import screen_spec
    from airline_market.sim.tape import ExogenousTape
    from airline_market.solvers.lagged_dp import history_dp
    from airline_market.spec import compile_market

    c = compile_market(
        screen_spec(
            {
                "T": 2,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 2,
                "capacity_support": [1, 2],
                "price_scale": 1.417,
                "outside_utility_shift": 0.262,
            }
        )
    )
    generate = ExogenousTape.generate
    calls = []

    def observed(compiled, seed, registry=None):
        calls.append(seed)
        return generate(compiled, seed, registry)

    monkeypatch.setattr(ExogenousTape, "generate", observed)
    for arm in ["direct", "linked", "unrelated", "oracle", "full_state"]:
        dp = history_dp(c, "A", arm, max_nodes=0)
        assert dp.finite is not None
    assert calls == [0] * 5  # object construction only: no solve, traversal or cohort
