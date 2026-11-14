"""Prospective seed-sealing checks; development artifacts remain development."""

import pytest

from airline_market.eval.seeds import SeedPlan
from airline_market.eval.sealing import validate_seed_release
from airline_market.eval.sealing import audit_manifest_sealing


def test_reserved_split_is_unavailable_to_development():
    plan = SeedPlan(tuple(range(100000, 100004)), (200000, 200001), (300000, 300001))
    with pytest.raises(ValueError, match="sealed"):
        validate_seed_release(plan)
    validate_seed_release(plan, "final", check_freeze=False)


def test_final_split_cannot_hide_test_seeds_in_training():
    plan = SeedPlan((300001,), (200000,), (300002, 300003))
    with pytest.raises(ValueError, match="split ranges"):
        validate_seed_release(plan, "final", check_freeze=False)


def test_existing_development_and_validation_cohorts_remain_valid():
    validate_seed_release(SeedPlan((100000,), (200000,), (200001, 200002)))
    validate_seed_release(SeedPlan((100,), (1000,), (10000, 10001)))


def test_registered_results_have_no_premature_test_release():
    assert audit_manifest_sealing("results") == []


def test_manifest_scan_detects_premature_opening(tmp_path):
    import json

    output = tmp_path / "bad_run"
    output.mkdir()
    (output / "results.json").write_text(
        json.dumps({"phase": "development", "seeds": {"test": [300000, 300001]}})
    )
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"output": str(output)}) + "\n")
    assert audit_manifest_sealing(tmp_path)[0]["reserved_seed_count"] == 2
