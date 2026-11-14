"""Register a common price-action class for exact information-lag controls."""

import json
from collections import defaultdict
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json
from airline_market.spec import compile_market, dump_spec, load_spec


def prepare():
    """Transform frozen T4/A4 sources, preserving original solves and physical laws."""
    source_config = json.loads(Path("config/campaigns/F1_history_anchors.json").read_text())
    reused = json.loads(Path("results/F1_history_tooling/reused_lag0.json").read_text())
    records = [{"parameters": r["parameters"], "source_spec": r["source_spec"],
                "source_spec_sha256": r["source_spec_sha256"]} for r in reused["records"]]
    records += [{"parameters": c, "source_spec": c["source_spec"],
                 "source_spec_sha256": c["source_spec_sha256"]} for c in source_config["cells"]]
    out = Path("results/common_price_lag_tooling/source_specs")
    out.mkdir(parents=True, exist_ok=True)
    quote_features = {"direct_rival_gap", "min_rival_gap", "linked_scarcity_signal", "rival_availability"}
    cells, manifest, projections = [], [], defaultdict(set)
    for record in records:
        source = Path(record["source_spec"])
        if file_hash(source) != record["source_spec_sha256"]:
            raise ValueError("original lag-control source changed")
        data = json.loads(source.read_text())
        original = compile_market(load_spec(data))
        p = record["parameters"]
        focal = data["metadata"].get("focal", "A")
        pricing = next(x for x in data["pricing"] if x["carrier"] == focal)
        removed = [x for x in pricing["fixed_features"] if x[0]["name"] in quote_features]
        pricing["fixed_features"] = [x for x in pricing["fixed_features"] if x[0]["name"] not in quote_features]
        if any(f["name"] in quote_features for f in pricing["controllable_features"]):
            raise ValueError("quote-dependent controllable basis needs a separate control design")
        data["metadata"]["lag_price_rule_control"] = {
            "scope": "common finite own-state price class across lag0/1; original models preserved",
            "removed_fixed_features": removed}
        revised = compile_market(load_spec(data))
        if original.A_global.tolist() != revised.A_global.tolist():
            raise ValueError("lag price control changed physical incidence")
        projection = {k: v for k, v in data.items() if k not in ("metadata", "name", "quotes", "information")}
        projections[p["connecting_share"], p["rival_capacity_prior"]].add(canonical_json(projection))
        target = out / (record["source_spec_sha256"][:16] + ".json")
        text = dump_spec(revised.spec)
        if target.exists() and target.read_text() != text:
            raise ValueError("existing common-rule source differs")
        if not target.exists():
            target.write_text(text)
        cell = {k: p[k] for k in ["instance", "n_spokes", "connecting_share",
            "rival_capacity_prior", "lag_periods", "T", "actions", "arm"]}
        cell.update(backend="compact_history_lag1", reservation_gb=1,
                    source_spec=str(target), source_spec_sha256=file_hash(target))
        cells.append(cell)
        manifest.append({"original_source": str(source), "original_sha256": file_hash(source),
            "controlled_source": str(target), "controlled_sha256": file_hash(target),
            "removed_fixed_features": removed, "parameters": cell})
    if len(cells) != 40 or any(len(values) != 1 for values in projections.values()):
        raise ValueError("common price-rule controls do not preserve all eight matched corners")
    config = {"schema": "airline-campaign/1", "id": "V3_common_price_lag",
        "phase": 2, "implementation_revision": "history-common-price-rule-v1",
        "task": "study:airline_market.experiments.history_study:solve", "design": "list",
        "cells": cells, "workers": 1, "memory_budget_gb": 1,
        "limits": {"seconds": 14400, "rss_gb": 1, "max_nodes": 20000000},
        "out": "results/V3/20261112_common_price_lag_T4A4_v1",
        "note": "Forty new T4/A4 common-price-rule models, not reruns of solved original sources. Remove quote-dependent fixed price features while preserving all physical parameters and the already restricted coefficient grid. Test within-corner information order and descriptive fresh/stale contrasts on a common own-state price class. Expected oracle/full-state lag contrast is zero; preserve violations and limits. This is a new V3/MR9 control, not a replacement frontier or a largest-scale selection; lag2 and full V3 remain unfinished."}
    atomic_json(Path("config/campaigns/V3_common_price_lag.json"), config)
    report = {"configured_cells": len(cells), "matched_physical_price_projections": len(projections),
        "physical_and_price_projection_identical_across_lags_and_arms": True,
        "original_sources_unchanged": True, "models_run": False, "records": manifest}
    atomic_json(Path("results/common_price_lag_tooling/registration.json"), report)
    return {k: v for k, v in report.items() if k != "records"}


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
