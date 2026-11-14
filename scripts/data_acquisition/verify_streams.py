"""Stream downloaded tables, skipping ZIP documentation without discarding it."""

from pathlib import Path
import argparse
import json
import re
import zipfile
import sys

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).parent))
from download import ROOT, RECORDS, atomic, inventory, now


def norm(name):
    """Match literal source field spellings while retaining original headers."""
    return re.sub("[^a-z0-9]", "", name.lower())


def stream_table(binary, headers, source):
    """Count rows and actual coverage in bounded batches with one CPU thread."""
    bynorm = {norm(name): name for name in headers if name}
    keys = (
        "year",
        "quarter",
        "month",
        "rpyear",
        "rpmonth",
        "schflyr",
        "schflmo",
        "fldate",
        "flightdate",
        "date",
        "searchdate",
        "startingairport",
        "destinationairport",
        "station",
    )
    selected = list(dict.fromkeys(bynorm[k] for k in keys if k in bynorm))
    if not selected:
        selected = [next(h for h in headers if h)]
    reader = pacsv.open_csv(
        binary,
        read_options=pacsv.ReadOptions(block_size=4 * 1024**2, use_threads=False),
        convert_options=pacsv.ConvertOptions(
            include_columns=selected, column_types={h: pa.string() for h in selected}
        ),
    )
    count, ranges, airports, periods = 0, {}, set(), {}
    for batch in reader:
        count += batch.num_rows
        for name in selected:
            column = batch.column(batch.schema.get_field_index(name))
            stat = pc.min_max(column).as_py()
            low, high = stat["min"], stat["max"]
            if low is not None and low != "":
                old = ranges.get(name, [low, high])
                ranges[name] = [min(old[0], low), max(old[1], high)]
            key = norm(name)
            if key in ("startingairport", "destinationairport"):
                airports.update(pc.unique(column).to_pylist())
            if key in (
                "year",
                "quarter",
                "month",
                "rpyear",
                "rpmonth",
                "schflyr",
                "schflmo",
            ):
                periods.setdefault(name, set()).update(pc.unique(column).to_pylist())
    exactdate = next(
        (name for name in selected if norm(name) in ("flightdate", "fldate", "date")),
        None,
    )
    return {
        "rows": count,
        "column_ranges": ranges,
        "date_range": ranges.get(exactdate) if exactdate else None,
        "observed_period_values": {
            k: sorted(v for v in vals if v) for k, vals in periods.items()
        },
        "airports": sorted(a for a in airports if a),
        "full_stream_crc_verified": True,
    }


def verify(receipt_path):
    """Add raw validation evidence; never label an absent loader as verified."""
    record = json.loads(Path(receipt_path).read_text())
    if record.get("status") != "downloaded":
        return {"id": record["id"], "status": "not_downloaded"}
    path = ROOT / record["path"]
    evidence = RECORDS / "validation" / (record["id"] + ".json")
    if evidence.exists():
        old = json.loads(evidence.read_text())
        if (
            old.get("sha256") == record["sha256"]
            and old.get("status") == "raw_stream_verified"
        ):
            return old
    members = []
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.filename.lower().endswith(".csv"):
                    continue
                header = next(
                    (
                        m.get("header", [])
                        for m in record.get("members", [])
                        if m["member"] == info.filename
                    ),
                    [],
                )
                if info.filename.lower().split("/")[-1] == "documentation.csv":
                    members.append(
                        {
                            "member": info.filename,
                            "role": "source_field_documentation",
                            "excluded_from_data_row_count": True,
                        }
                    )
                    continue
                with archive.open(info) as stream:
                    members.append(
                        {
                            "member": info.filename,
                            "headers": header,
                            **stream_table(stream, header, record["source"]),
                        }
                    )
    elif path.suffix == ".csv":
        header = record["members"][0]["header"]
        with path.open("rb") as stream:
            members.append(
                {
                    "member": path.name,
                    "headers": header,
                    **stream_table(stream, header, record["source"]),
                }
            )
    result = {
        "id": record["id"],
        "status": "raw_stream_verified" if members else "not_tabular",
        "sha256": record["sha256"],
        "path": record["path"],
        "verified_at": now(),
        "members": members,
        "rows": sum(m.get("rows", 0) for m in members),
        "loader_status": "Not inferred from raw streaming. Existing supported loader maps checked separately; DB1C, marketing attribution and new cost/inventory sources need their declared loaders.",
    }
    atomic(evidence, result)
    print(record["id"], result["status"], result["rows"], flush=True)
    return result


def main():
    """Verify matching receipts once; preserve failures and continue other files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", action="append")
    args = parser.parse_args()
    receipts = sorted((RECORDS / "files").glob("*.json"))
    for receipt in receipts:
        if args.id and receipt.stem not in args.id:
            continue
        try:
            verify(receipt)
        except Exception as error:
            atomic(
                RECORDS / "validation" / (receipt.stem + ".json"),
                {
                    "id": receipt.stem,
                    "status": "raw_validation_failed",
                    "reason": str(error),
                    "verified_at": now(),
                },
            )
            print(receipt.stem, "FAILED", error, flush=True)
    inventory()


if __name__ == "__main__":
    main()
