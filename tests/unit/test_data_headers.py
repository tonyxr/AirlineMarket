"""Real-file inventory behavior using local raw-format fixtures only."""

import json
from pathlib import Path
import zipfile

from airline_market.pipeline.verify import verify_headers


def test_zip_header_count_and_partial_inventory(tmp_path):
    path = tmp_path / "segments.zip"
    text = "YEAR,MONTH,UNIQUE_CARRIER,ORIGIN,DEST,SEATS,DEPARTURES_PERFORMED,PASSENGERS\n2022,1,AA,ORD,DEN,200,2,150\n2022,2,UA,ORD,DEN,300,3,200\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("real_format.csv", text)
    inventory = tmp_path / "inventory.json"
    result = verify_headers(path, "t100", count_rows=True, inventory=inventory)
    member = result["files"][0]["members"][0]
    assert member["row_count"] == 2
    assert member["date_range"] == ["2022-01", "2022-02"]
    assert member["unknown_required_columns"] == []
    assert (
        json.loads(inventory.read_text())["status"]
        == "partial_inventory_coverage_not_yet_reconciled"
    )
