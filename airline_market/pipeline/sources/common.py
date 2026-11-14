from pathlib import Path
import hashlib, json
from ...core.errors import DataValidationError
from ..fieldmap import FieldMap, MAPS


def stream(scope, source, required, *, chunksize=100000, out=None):
    """
    Read/filter source chunks and record real headers, row counts and hashes (model
    §§1–3, calibration assumptions).
    """
    import pandas as pd

    entry = scope.sources.get(source)
    paths = (
        [Path(entry)]
        if entry and Path(entry).is_file()
        else sorted(Path(entry or "/nonexistent").glob("*.csv*"))
    )
    if not paths:
        raise DataValidationError(f"no local {source} files for {scope.name}: {entry}")
    frames = []
    metadata = []
    for path in paths:
        count = kept = 0
        headers = None
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            headers = list(chunk)
            mapping = FieldMap(MAPS[source]).columns(headers, required)
            chunk = chunk.rename(columns=mapping)
            count += len(chunk)
            carrier = (
                "carrier"
                if "carrier" in chunk
                else "ticket_carrier"
                if "ticket_carrier" in chunk
                else None
            )
            if carrier:
                chunk = chunk[chunk[carrier].isin(scope.carriers)]
            if source == "noaa" and scope.params.get("weather_stations"):
                chunk = chunk[
                    chunk.station.isin(scope.params["weather_stations"].values())
                ]
            if source == "flightprices":
                hubs = scope.params.get("airports", scope.hubs)
                relevant = chunk.startingAirport.isin(
                    hubs
                ) | chunk.destinationAirport.isin(hubs)
                if "segmentsArrivalAirportCode" in chunk:
                    relevant |= (
                        chunk.segmentsArrivalAirportCode.fillna("")
                        .str.split("||", regex=False)
                        .map(lambda xs: bool(set(xs) & set(hubs)))
                    )
                chunk = chunk[relevant]
                if "segmentsAirlineCode" in chunk:
                    chunk = chunk[
                        chunk.segmentsAirlineCode.fillna("")
                        .str.split("||", regex=False)
                        .map(lambda xs: bool(set(xs) & set(scope.carriers)))
                    ]
            # Retain feeder O&D demand; scope airports can narrow large quote panels.
            airports = scope.params.get("airports")
            if airports and ("origin" in chunk or "startingAirport" in chunk):
                o = "origin" if "origin" in chunk else "startingAirport"
                z = "dest" if "dest" in chunk else "destinationAirport"
                chunk = chunk[chunk[o].isin(airports) | chunk[z].isin(airports)]
            if source == "ontime" and scope.departure_date:
                chunk = chunk[
                    chunk.flight_date.astype(str).str[:10] == scope.departure_date
                ]
            kept += len(chunk)
            frames.append(chunk)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        metadata.append(
            {
                "path": str(path.resolve()),
                "sha256": digest.hexdigest(),
                "headers": headers,
                "rows_read": count,
                "rows_kept": kept,
            }
        )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frame.attrs["sources"] = metadata
    return persist(frame, scope, source, out)


def persist(frame, scope, source, out=None):
    """
    Write the normalized subset and source/filter provenance sidecar (model §§1–3,
    calibration assumptions).
    """
    if out is not False:
        dest = Path(out or f"data/processed/{scope.name}")
        dest.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dest / (source + ".parquet"), index=False)
        (dest / (source + ".json")).write_text(
            json.dumps(
                {
                    "sources": frame.attrs.get("sources", []),
                    "normalized_rows": len(frame),
                    "filters": {
                        "carriers": scope.carriers,
                        "airports": scope.params.get("airports"),
                        "departure_date": scope.departure_date,
                    },
                },
                indent=2,
            )
        )
    return frame
