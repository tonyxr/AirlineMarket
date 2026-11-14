"""S1: saved exact values, declared units, MR7/MR9 and rank-aware DSD fits.

All rows and signed outcomes survive. Missing/limited values stay null. This
module reads returned publisher futures only; it never solves or simulates.
"""

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.design import expand_campaign, load_campaign
from airline_market.campaign.settled_runner import completed_snapshot
from airline_market.experiments.screen_design import FACTORS

ARMS = ("direct", "linked", "unrelated", "oracle", "full_state")
FACTORS_SEVEN = (*FACTORS, "bottleneck", "lag")
RANK_TOLERANCE = 1e-10


def normalize_values(results, profit_scale, share_weight=0):
    """Dollar profit effects for pure-profit S1 sources, using saved gross revenue.

    MR tolerances apply to the original normalized objectives. G uses the
    unchanged inclusive 0.1%-of-direct-revenue rule, with no epsilon repair.
    """
    if not math.isfinite(profit_scale) or profit_scale <= 0 or share_weight != 0:
        raise ValueError(
            "S1 dollar-profit conversion requires positive scale and zero share weight"
        )
    values, statuses = {}, {}
    for arm in ARMS:
        result = results.get(arm)
        statuses[arm] = result["status"] if result is not None else "unstarted"
        value = result.get("value") if result is not None else None
        if statuses[arm] == "solved":
            if value is None or not math.isfinite(value) or not result["exact"]:
                raise ValueError("solved exact arm has no finite exact value")
            values[arm] = value
        else:
            if value is not None:
                raise ValueError(
                    "a limited or failed arm cannot supply a scientific value"
                )
            values[arm] = None
    direct = results.get("direct")
    diagnostics = direct.get("diagnostics", {}) if direct is not None else {}
    revenue = diagnostics.get("revenue_A") if statuses["direct"] == "solved" else None
    if revenue is not None and (not math.isfinite(revenue) or revenue < 0):
        raise ValueError("invalid saved direct gross revenue")
    effects = {}
    for name, arm in [
        ("delta_N", "linked"),
        ("delta_U", "unrelated"),
        ("oracle_gap", "oracle"),
        ("full_state_gap", "full_state"),
    ]:
        delta = (
            None
            if values[arm] is None or values["direct"] is None
            else values[arm] - values["direct"]
        )
        usd = None if delta is None else delta * profit_scale
        effects[name] = {
            "normalized": delta,
            "usd_per_cohort": usd,
            "percent_direct_revenue": None
            if usd is None or not revenue
            else 100 * usd / revenue,
        }
    oracle = effects["oracle_gap"]["usd_per_cohort"]
    eligible = bool(
        revenue is not None
        and revenue > 0
        and oracle is not None
        and oracle >= 0.001 * revenue
    )
    for effect in effects.values():
        effect["share_oracle_gap"] = (
            effect["usd_per_cohort"] / oracle
            if eligible and effect["usd_per_cohort"] is not None
            else None
        )
    mr7 = (
        None
        if effects["delta_U"]["normalized"] is None
        else abs(effects["delta_U"]["normalized"]) <= 1e-12
    )
    order = {}
    for left, right in zip(
        ("direct", "linked", "oracle"), ("linked", "oracle", "full_state")
    ):
        order[left + "<=" + right] = (
            None
            if values[left] is None or values[right] is None
            else values[left] <= values[right] + 1e-12
        )
    mr9 = False if False in order.values() else None if None in order.values() else True
    return {
        "values_normalized": values,
        "values_usd_per_cohort": {
            a: None if v is None else v * profit_scale for a, v in values.items()
        },
        "statuses": statuses,
        "effects": effects,
        "direct_gross_revenue_usd": revenue,
        "dollar_scope": "Synthetic native model net-profit expectation; no empirical-profit claim",
        "G_eligible": eligible,
        "G": effects["delta_N"]["share_oracle_gap"],
        "G_rule": "Oracle gap >=0.1% of positive direct gross revenue; otherwise null",
        "MR7": mr7,
        "MR9": mr9,
        "MR9_pairs": order,
        "plausible": diagnostics.get("plausible"),
    }


def coded_rows(settings):
    """Use the registered ordinal levels, not invented numerical prior distances."""
    rows = []
    for setting in settings:
        x, p = setting["coded"], setting["parameters"]
        if (
            len(x) != 5
            or any(v not in (-1, 0, 1) for v in x)
            or p["bottleneck"] not in ("rival", "both")
            or p["lag_periods"] not in (0, 1)
        ):
            raise ValueError("invalid registered S1 factor coordinates")
        rows.append(
            [*x, -1 if p["bottleneck"] == "rival" else 1, 2 * p["lag_periods"] - 1]
        )
    return np.asarray(rows, dtype=float).reshape(-1, 7)


def feature_matrix(coded, quadratic=False):
    """Primary: intercept + seven main effects + all 21 two-factor products."""
    names = ["intercept", *FACTORS_SEVEN]
    columns = [np.ones(len(coded)), *coded.T]
    for a, b in combinations(range(7), 2):
        names.append(FACTORS_SEVEN[a] + ":" + FACTORS_SEVEN[b])
        columns.append(coded[:, a] * coded[:, b])
    if quadratic:
        for a in range(5):
            names.append(FACTORS_SEVEN[a] + "^2")
            columns.append(coded[:, a] ** 2)
    return np.column_stack(columns), names


def matrix_audit(matrix, names):
    """Report null directions and individual-coefficient estimability explicitly."""
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = (
        int(np.count_nonzero(singular > singular[0] * RANK_TOLERANCE))
        if len(singular) and singular[0]
        else 0
    )
    null = vh[rank:]
    error = np.linalg.norm(null, axis=0)
    return {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "rank": rank,
        "nullity": matrix.shape[1] - rank,
        "relative_svd_tolerance": RANK_TOLERANCE,
        "singular_values": singular.tolist(),
        "null_basis": null.tolist(),
        "terms": names,
        "estimable": dict(zip(names, map(bool, error <= RANK_TOLERANCE))),
        "estimability_error": dict(zip(names, map(float, error))),
    }


def fit_response(settings, responses, *, quadratic=False):
    """Descriptive least-squares projection; unidentified coefficients stay null.

    Exact model expectations are not Monte Carlo estimates. No p-values or
    sampling intervals are invented from deterministic response residuals.
    """
    ids = [s["setting_id"] for s in settings]
    if len(set(ids)) != len(ids) or set(responses) - set(ids):
        raise ValueError("duplicate or unknown S1 setting")
    coded = coded_rows(settings)
    all_x, names = feature_matrix(coded, quadratic)
    used = [i for i, key in enumerate(ids) if responses.get(key) is not None]
    y = np.asarray([responses[ids[i]] for i in used], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("nonfinite regression response")
    x = all_x[used]
    audit = matrix_audit(x, names)
    beta = np.linalg.lstsq(x, y, rcond=RANK_TOLERANCE)[0]
    predictions = x @ beta
    residual = y - predictions
    total_ss = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0
    null = np.asarray(audit["null_basis"]).reshape(-1, len(names))
    partial = {}
    for factor in ("connecting_share", "price_scale"):
        j = FACTORS_SEVEN.index(factor)
        rows = []
        for level in (-1, 0, 1):
            changed = coded.copy()
            changed[:, j] = level
            average = feature_matrix(changed, quadratic)[0].mean(axis=0)
            error = float(np.linalg.norm(null @ average))
            rows.append(
                {
                    "coded_level": level,
                    "physical_level": FACTORS[factor][level + 1],
                    "value": float(average @ beta)
                    if len(used) and error <= RANK_TOLERANCE
                    else None,
                    "estimability_error": error,
                }
            )
        partial[factor] = rows
    return {
        "status": "fit" if used else "pending",
        "design": audit,
        "used_settings": [ids[i] for i in used],
        "missing_settings": [key for key in ids if responses.get(key) is None],
        "coefficients": {
            name: float(beta[j]) if audit["estimable"][name] else None
            for j, name in enumerate(names)
        },
        "fitted": {
            ids[i]: {
                "response": float(y[k]),
                "fitted": float(predictions[k]),
                "residual": float(residual[k]),
            }
            for k, i in enumerate(used)
        },
        "R_squared": None
        if not total_ss
        else 1 - float(residual @ residual) / total_ss,
        "partial_dependence_empirical_design_average": partial,
        "interpretation": "Coded-level descriptive projection; null coefficients are not individually identified; no inferential p-values",
    }


def campaign_summary(config_path, root):
    """Retain every registered setting, reading only acknowledged terminal cells."""
    config_path, root = Path(config_path), Path(root)
    config = load_campaign(config_path)
    if config["id"] != "S1-r1":
        raise ValueError(
            "this report implements the owner-retained S1 round-one design only"
        )
    cells = expand_campaign(config)
    acknowledged = (
        set(json.loads((root / "settled_cells.json").read_bytes())["cell_ids"])
        if (root / "settled_cells.json").exists()
        else set()
    )
    if acknowledged - {c["cell_id"] for c in cells}:
        raise ValueError("unknown acknowledged cell")
    groups = {}
    for cell in cells:
        p = cell["parameters"]
        key = p["setting_id"]
        identity = {
            k: p[k]
            for k in (
                "coded",
                "dsd_row",
                "screen_parameters",
                "source_spec",
                "source_spec_sha256",
            )
        }
        group = groups.setdefault(key, {"identity": identity, "cells": {}})
        if group["identity"] != identity or p["arm"] in group["cells"]:
            raise ValueError("inconsistent or duplicate arm registration")
        group["cells"][p["arm"]] = cell
    if len(groups) != 52 or len(cells) != 260:
        raise ValueError("S1 round one must retain all52 settings and260 arm jobs")
    settings, summaries = [], []
    for key, group in groups.items():
        if set(group["cells"]) != set(ARMS):
            raise ValueError("registered setting is missing an arm")
        identity = group["identity"]
        source = Path(identity["source_spec"])
        if file_hash(source) != identity["source_spec_sha256"]:
            raise ValueError("registered physical source changed")
        spec = json.loads(source.read_bytes())
        objective = next(o for o in spec["objectives"] if o["carrier"] == "A")
        settings.append(
            {
                "setting_id": key,
                "coded": identity["coded"],
                "parameters": identity["screen_parameters"],
            }
        )
        results, provenance = {}, {}
        for arm, cell in group["cells"].items():
            if cell["cell_id"] not in acknowledged:
                continue
            found = completed_snapshot(
                root / "cells" / cell["cell_id"], cell["config_hash"]
            )
            if found is None:
                raise ValueError("acknowledged cell has no terminal result")
            directory, result = found
            if result["parameters"] != cell["parameters"]:
                raise ValueError("saved cell parameters differ")
            results[arm] = result
            provenance[arm] = {
                "cell_id": cell["cell_id"],
                "config_hash": cell["config_hash"],
                "results_sha256": file_hash(directory / "results.json"),
                "git_sha": json.loads((directory / "environment.json").read_bytes())[
                    "git_sha"
                ],
            }
        summary = normalize_values(
            results, objective["profit_scale"], objective["share_weight"]
        )
        # Actual diagnostics and the owner's structural exclusion are distinct.
        summary.update(
            setting_id=key,
            source_sha256=identity["source_spec_sha256"],
            provenance=provenance,
            structurally_eligible=False,
            eligibility_reason="Owner retained all52 despite the unchanged two-carrier70pct load bound",
        )
        summaries.append(summary)
    delta = {
        s["setting_id"]: s["effects"]["delta_N"]["percent_direct_revenue"]
        for s in summaries
    }
    logs, omitted = {}, {}
    for row in summaries:
        gap = row["effects"]["oracle_gap"]["percent_direct_revenue"]
        if gap is not None and gap > 0:
            logs[row["setting_id"]] = math.log(gap)
        else:
            omitted[row["setting_id"]] = (
                "unmeasured"
                if gap is None
                else "nonpositive gap; no log epsilon or deletion from value table"
            )
    return {
        "plan_id": "S1",
        "config_sha256": file_hash(config_path),
        "settings": summaries,
        "registered_settings": len(settings),
        "registered_arm_jobs": len(cells),
        "acknowledged_cells": len(acknowledged),
        "primary_design": matrix_audit(*feature_matrix(coded_rows(settings))),
        "quadratic_design_sensitivity": matrix_audit(
            *feature_matrix(coded_rows(settings), True)
        ),
        "delta_N_percent_regression": fit_response(settings, delta),
        "log_oracle_gap_percent_regression": fit_response(settings, logs),
        "log_gap_omissions": omitted,
        "log_response_definition": "Natural log of oracle gap in percent of direct gross revenue; nonpositive or missing outcomes stay untransformed in the value table",
        "plausible_ranking": [],
        "best_plausible_setting": None,
        "D1_case": None,
        "selection_status": "No structurally eligible regime; this is not evidence of small/zero gaps and does not establish caseC",
        "new_solver_or_simulator_calls": 0,
        "scientific_budget": None,
    }
