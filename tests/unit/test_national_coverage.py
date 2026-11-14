"""Local sequence fixtures for national-to-core coupon-edge attribution."""

import json
from types import SimpleNamespace
import pandas as pd
import pytest

from airline_market.pipeline.national_coverage import COLUMNS, classify_market, audit_national
from airline_market.campaign.artifacts import atomic_json

SCOPE = {"carriers": ["UA", "AA"], "hub_candidate": "ORD", "airport_pool": ["ORD", "BOS", "JFK"]}
ROUTES = {("UA", "ORD", "BOS"), ("UA", "BOS", "ORD"),
          ("AA", "BOS", "ORD"), ("UA", "ORD", "JFK")}
SELECTED = {"ORD-BOS", "BOS-ORD", "BOS-JFK"}


def row(path, ticket, operator=None, passengers=10):
    """Build one complete source-style directional market record."""
    airports = path.split(":")
    return {"ItinID": "1", "MktID": "10", "Year": "2022", "Quarter": "2",
            "Origin": airports[0], "Dest": airports[-1], "AirportGroup": path,
            "MktCoupons": str(len(airports) - 1), "TkCarrierGroup": ticket,
            "OpCarrierGroup": operator or ticket, "RPCarrier": "UA",
            "Passengers": str(passengers), "ItinGeoType": "1", "MktGeoType": "1"}


def test_off_pool_and_mixed_paths_keep_actual_ordered_core_edges():
    a = classify_market(row("ALB:ORD:BOS", "UA:UA", "YX:UA"), SCOPE, SELECTED, ROUTES)
    assert a["category"] == "off_pool_od"
    assert a["core"] == (("UA", "ORD", "BOS"),)
    b = classify_market(row("BOS:ORD:JFK", "AA:UA", "MQ:UA"), SCOPE, SELECTED, ROUTES)
    assert b["category"] == "in_pool_unrepresented_path"
    assert b["core"] == (("AA", "BOS", "ORD"), ("UA", "ORD", "JFK"))
    assert b["ticketing_class"] == "mixed_or_other"


def test_unknown_ticket_and_malformed_sequences_are_not_inferred_as_network_load():
    x = classify_market(row("ORD:BOS", "--", "UA"), SCOPE, SELECTED, ROUTES)
    assert x["core"] == () and x["unknown_ticket_edges"] == (("UA", "ORD", "BOS"),)
    malformed = row("ORD:BOS", "UA:UA")
    assert classify_market(malformed, SCOPE, SELECTED, ROUTES)["reason"] == "malformed_ordered_sequence"
    missing = row("ORD:BOS", "UA"); missing["TkCarrierGroup"] = None
    assert classify_market(missing, SCOPE, SELECTED, ROUTES)["reason"] == "missing_ordered_sequence"


def test_repeated_coupon_edges_keep_multiplicity_and_unselected_scope():
    x = classify_market(row("ORD:BOS:ORD:BOS", "UA:UA:UA"), SCOPE, SELECTED, ROUTES)
    assert x["repeated_core_route"] and len(x["core"]) == 3
    assert x["category"] == "in_pool_unrepresented_path"
    x = classify_market(row("BOS:ORD:JFK", "UA:UA"), SCOPE, {"ORD-BOS"}, ROUTES)
    assert x["category"] == "in_pool_unselected_market"


def test_bounded_source_audit_conserves_tenfold_coupon_counts_and_preserves_residual(tmp_path):
    scope = tmp_path / "scope.json"; atomic_json(scope, SCOPE)
    selected = tmp_path / "selected.csv"; pd.DataFrame({"market": sorted(SELECTED)}).to_csv(selected, index=False)
    cap = tmp_path / "capacity.csv"
    pd.DataFrame([{"year": 2022, "month": 4, "marketing_carrier": c, "origin": o,
                   "dest": d, "allocated_passengers": 1000} for c, o, d in sorted(ROUTES)]).to_csv(cap, index=False)
    source = tmp_path / "q2.csv"
    rows = [row("ORD:BOS", "UA", passengers=10),
            row("ALB:ORD:BOS", "UA:UA", passengers=20),
            row("BOS:ORD:ALB", "UA:UA", passengers=30),
            row("BOS:ORD:JFK", "AA:UA", passengers=5),
            row("ORD:BOS", "--", "UA", passengers=1)]
    pd.DataFrame(rows)[COLUMNS].to_csv(source, index=False)
    result = audit_national({"scope": str(scope), "selected_markets": str(selected),
        "capacity_targets": str(cap), "db1b_q2_archive": str(source)}, SimpleNamespace(directory=tmp_path))
    flows = pd.read_csv(tmp_path / "route_category_flows.csv")
    assert flows.expanded_coupon_edge_passengers.sum() == 700
    assert flows.sample_passengers.sum() == 70
    ua = next(r for r in result["carrier_coverage"] if r["carrier"] == "UA")
    assert ua["national_expanded_coupon_edges"] == 650
    assert ua["allocated_T100_passengers"] == 3000
    assert ua["national_coupon_minus_T100"] == -2350
    assert result["counts_rescaled"] is False and result["background_process_fitted"] is False
    assert json.loads((tmp_path / "unknown_ticket_edges.json").read_text())[0]["attributed_to_network"] is False
    rows[0]["Quarter"] = "3"
    pd.DataFrame(rows)[COLUMNS].to_csv(source, index=False)
    with pytest.raises(ValueError, match="non-Q2"):
        audit_national({"scope": str(scope), "selected_markets": str(selected),
            "capacity_targets": str(cap), "db1b_q2_archive": str(source)}, SimpleNamespace(directory=tmp_path))
