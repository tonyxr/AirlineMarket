"""S1 value units, absent regimes and aliased deterministic design projections."""

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.design import expand_campaign
from airline_market.experiments.screen_analysis import (
    ARMS,
    campaign_summary,
    coded_rows,
    feature_matrix,
    fit_response,
    matrix_audit,
    normalize_values,
)
from airline_market.experiments.screen_design import round_one


def arms(values=(1, 1.002, 1, 1.01, 1.012), revenue=1000):
    return {
        a: {
            "status": "solved",
            "value": v,
            "exact": True,
            "diagnostics": {"revenue_A": revenue, "plausible": False},
        }
        for a, v in zip(ARMS, values)
    }


def test_profit_units_revenue_denominator_and_native_mr_tolerances():
    out = normalize_values(arms(), 1000)
    assert out["effects"]["delta_N"]["usd_per_cohort"] == pytest.approx(2)
    assert out["effects"]["delta_N"]["percent_direct_revenue"] == pytest.approx(0.2)
    assert out["G"] == pytest.approx(0.2)
    assert out["MR7"] is out["MR9"] is True


@pytest.mark.parametrize("placebo,passed", [(5e-13, True), (2e-12, False)])
def test_MR_checks_use_original_normalized_tolerance(placebo, passed):
    out = normalize_values(arms((0, -5e-13, placebo, 0, 0)), 1000)
    assert out["MR7"] is passed and out["MR9"] is True
    assert out["effects"]["delta_N"]["normalized"] == -5e-13


@pytest.mark.parametrize(
    "gap,expected", [(1, True), (0.999, False), (0, False), (-1, False)]
)
def test_G_has_unchanged_inclusive_threshold_without_gap_repair(gap, expected):
    out = normalize_values(arms((0, 0.1, 0, gap, 2), revenue=1000), 1)
    assert out["G_eligible"] is expected
    assert (out["G"] is not None) is expected


def test_zero_revenue_and_negative_results_are_kept():
    out = normalize_values(arms((1, 0.9, 1.1, 1.2, 1.3), revenue=0), 1000)
    assert out["effects"]["delta_N"]["usd_per_cohort"] == pytest.approx(-100)
    assert out["effects"]["delta_N"]["percent_direct_revenue"] is None
    assert out["G"] is None and out["MR7"] is out["MR9"] is False


def test_limit_blocks_only_dependent_effects():
    data = arms()
    data["oracle"] = {"status": "memory_limit", "value": None}
    out = normalize_values(data, 1000)
    assert out["effects"]["delta_N"]["usd_per_cohort"] == pytest.approx(2)
    assert out["effects"]["oracle_gap"]["usd_per_cohort"] is None
    assert out["G"] is None and out["MR9"] is None and out["MR7"] is True
    data["oracle"]["value"] = 8
    with pytest.raises(ValueError, match="limited"):
        normalize_values(data, 1000)


def test_share_bonus_is_not_mislabeled_as_dollar_profit():
    with pytest.raises(ValueError, match="zero share"):
        normalize_values(arms(), 1000, share_weight=0.1)


def test_registered_design_rank_nullspace_and_unique_main_effects():
    settings = round_one()
    for quadratic, rank, columns in [(False, 24, 29), (True, 25, 34)]:
        matrix, names = feature_matrix(coded_rows(settings), quadratic)
        audit = matrix_audit(matrix, names)
        assert audit["rank"] == rank and audit["columns"] == columns
        assert audit["nullity"] == columns - rank
        assert all(audit["estimable"][name] for name in names[:8])
        assert not audit["estimable"]["connecting_share:arrival_probability"]
        assert np.max(np.abs(matrix @ np.array(audit["null_basis"]).T)) < 1e-12


def test_fit_recovers_identified_effects_without_fabricating_aliased_coefficients():
    settings = round_one()
    coded = coded_rows(settings)
    y = 1 + 2 * coded[:, 0] - 3 * coded[:, 5] + 4 * coded[:, 4] * coded[:, 6]
    fit = fit_response(settings, dict(zip([s["setting_id"] for s in settings], y)))
    assert fit["coefficients"]["connecting_share"] == pytest.approx(2)
    assert fit["coefficients"]["bottleneck"] == pytest.approx(-3)
    assert fit["coefficients"]["price_scale:lag"] == pytest.approx(4)
    assert fit["coefficients"]["connecting_share:arrival_probability"] is None
    assert fit["R_squared"] == pytest.approx(1)
    assert max(abs(r["residual"]) for r in fit["fitted"].values()) < 1e-12
    assert [
        p["value"]
        for p in fit["partial_dependence_empirical_design_average"]["connecting_share"]
    ] == pytest.approx([-1, 1, 3])


def test_missing_responses_retain_registration_and_estimability_limits():
    settings = round_one()
    empty = fit_response(settings, {})
    assert empty["status"] == "pending" and len(empty["missing_settings"]) == 52
    partial = fit_response(settings, {settings[0]["setting_id"]: -2})
    assert len(partial["missing_settings"]) == 51
    assert partial["fitted"][settings[0]["setting_id"]]["response"] == -2
    assert partial["coefficients"]["connecting_share"] is None
    with pytest.raises(ValueError, match="unknown"):
        fit_response(settings, {"other": 0})


def test_file_join_uses_acknowledged_cells_and_keeps_no_eligible_regime(tmp_path):
    settings = round_one()
    rows = []
    for s in settings:
        source = tmp_path / (s["setting_id"] + ".json")
        atomic_json(
            source,
            {
                "objectives": [
                    {"carrier": "A", "profit_scale": 1000, "share_weight": 0}
                ],
                "fixture_id": s["setting_id"],
            },
        )
        for arm in ARMS:
            rows.append(
                {
                    "setting_id": s["setting_id"],
                    "coded": s["coded"],
                    "dsd_row": s["dsd_row"],
                    "screen_parameters": s["parameters"],
                    "source_spec": str(source),
                    "source_spec_sha256": file_hash(source),
                    "arm": arm,
                }
            )
    config = {
        "schema": "airline-campaign/1",
        "id": "S1-r1",
        "design": "list",
        "cells": rows,
    }
    path = tmp_path / "config.json"
    atomic_json(path, config)
    out = campaign_summary(path, tmp_path)
    assert out["registered_settings"] == 52 and out["registered_arm_jobs"] == 260
    assert out["acknowledged_cells"] == 0 and len(out["settings"]) == 52
    cells = expand_campaign(config)[:5]
    for cell, result in zip(cells, arms((1, 0.9, 1, 1, 1)).values()):
        directory = tmp_path / "cells" / cell["cell_id"] / "attempt-0001"
        directory.mkdir(parents=True)
        result.update(config_hash=cell["config_hash"], parameters=cell["parameters"])
        atomic_json(directory / "results.json", result)
        atomic_json(
            directory / "COMPLETE.json",
            {
                "config_hash": cell["config_hash"],
                "status": result["status"],
                "results_sha256": file_hash(directory / "results.json"),
            },
        )
        atomic_json(directory / "environment.json", {"git_sha": "local-fixture"})
    # Complete files alone are not the coordinator's future-return acknowledgement.
    assert campaign_summary(path, tmp_path)["acknowledged_cells"] == 0
    atomic_json(
        tmp_path / "settled_cells.json", {"cell_ids": [c["cell_id"] for c in cells]}
    )
    out = campaign_summary(path, tmp_path)
    first = out["settings"][0]
    assert first["effects"]["delta_N"]["usd_per_cohort"] == pytest.approx(-100)
    assert (
        first["MR9"] is False
        and "nonpositive" in out["log_gap_omissions"][first["setting_id"]]
    )
    assert (
        out["best_plausible_setting"] is out["D1_case"] is None
        and out["plausible_ranking"] == []
    )
    assert len(out["settings"]) == 52 and out["new_solver_or_simulator_calls"] == 0
    source = tmp_path / (settings[-1]["setting_id"] + ".json")
    source.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        campaign_summary(path, tmp_path)
