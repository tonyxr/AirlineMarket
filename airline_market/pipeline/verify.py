"""Streaming real-source header/inventory verification (validation plan R0.5)."""

from contextlib import contextmanager
from pathlib import Path
import csv
import gzip
import io
import json
import zipfile

from ..campaign.artifacts import atomic_json, file_hash, utc_now
from .fieldmap import FieldMap, MAPS


REQUIRED = {
    "t100": (
        "carrier",
        "origin",
        "dest",
        "seats",
        "departures_performed",
        "passengers",
    ),
    "db1b": ("origin", "dest", "passengers", "market_fare", "market_coupons"),
    "coupon": ("itin_id", "origin", "dest"),
    "ontime": (
        "flight_date",
        "carrier",
        "flight_number",
        "origin",
        "dest",
        "cancelled",
    ),
    "flightprices": (
        "searchDate",
        "flightDate",
        "startingAirport",
        "destinationAirport",
        "totalFare",
    ),
    "noaa": ("station", "date"),
}


@contextmanager
def members(path: Path):
    """Open actual CSV members using bounded streaming rather than extraction."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not names:
                raise ValueError(f"archive has no CSV member: {path}")
            yield [(name, archive.open(name)) for name in names]
    else:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as handle:
            yield [(path.name, handle)]


def verify_headers(
    path: str | Path,
    source: str,
    *,
    count_rows: bool = False,
    inventory: str | Path | None = None,
) -> dict:
    """Verify required aliases and optionally count rows/date coverage on disk."""
    source = {"db1b_market": "db1b"}.get(source, source)
    if source not in REQUIRED:
        raise ValueError(
            f"no verified loader map for {source}; provide the actual header before implementing it"
        )
    path = Path(path).resolve()
    files = (
        sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.name.lower().endswith((".csv", ".csv.gz", ".zip"))
        )
        if path.is_dir()
        else [path]
    )
    records = []
    for filename in files:
        record = {
            "path": str(filename),
            "source": source,
            "bytes": filename.stat().st_size,
            "sha256": file_hash(filename),
            "verified_at": utc_now(),
            "members": [],
        }
        with members(filename) as entries:
            for name, binary in entries:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    rows = csv.reader(text)
                    headers = next(rows)
                    mapping = FieldMap(MAPS[source]).columns(headers, REQUIRED[source])
                    body = {
                        "name": name,
                        "headers": headers,
                        "mapping": mapping,
                        "unknown_required_columns": [],
                        "row_count": None,
                        "date_range": None,
                    }
                    if count_rows:
                        dates = []
                        date_name = next(
                            (
                                h
                                for h in headers
                                if mapping.get(h)
                                in ("flight_date", "flightDate", "date")
                            ),
                            None,
                        )
                        date_index = headers.index(date_name) if date_name else None
                        year = next(
                            (
                                headers.index(h)
                                for h in headers
                                if mapping.get(h) == "year"
                            ),
                            None,
                        )
                        month = next(
                            (
                                headers.index(h)
                                for h in headers
                                if mapping.get(h) == "month"
                            ),
                            None,
                        )
                        count = 0
                        low = high = None
                        for row in rows:
                            if not row:
                                continue
                            count += 1
                            date = (
                                row[date_index][:10]
                                if date_index is not None
                                else f"{int(row[year]):04d}-{int(row[month]):02d}"
                                if year is not None and month is not None
                                else None
                            )
                            if date:
                                low = date if low is None else min(low, date)
                                high = date if high is None else max(high, date)
                        body.update(
                            row_count=count,
                            date_range=[low, high] if low is not None else None,
                        )
                    record["members"].append(body)
        records.append(record)
    result = {
        "schema": "airline-source-verification/1",
        "status": "headers_and_rows_verified"
        if count_rows
        else "headers_verified_rows_not_counted",
        "files": records,
    }
    if inventory:
        target = Path(inventory)
        data = (
            json.loads(target.read_text())
            if target.exists()
            else {"schema": "airline-data-inventory/1", "files": []}
        )
        existing = {row["path"]: row for row in data["files"]}
        existing.update({row["path"]: row for row in records})
        data.update(
            files=[existing[key] for key in sorted(existing)], updated_at=utc_now()
        )
        # Presence alone does not establish the plan's required year/month coverage.
        data["status"] = "partial_inventory_coverage_not_yet_reconciled"
        atomic_json(target, data)
    return result
