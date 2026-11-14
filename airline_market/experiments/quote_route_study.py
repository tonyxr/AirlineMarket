"""New prepared-key route binding; no original quote ingestion or parameter fit."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.pipeline.quote_route_binding import BindingIndex, bind_reported


class Sink:
    """Append new derived evidence in bounded buffers, without touching inputs."""

    def __init__(self, path):
        self.path = path
        self.writer = None

    def write(self, rows):
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)

    def close(self):
        if self.writer:
            self.writer.close()


def run(cell, context):
    p = cell["parameters"]
    guards = [
        (p[n], p[n + "_sha256"])
        for n in [
            "quote_db",
            "legs",
            "products",
            "network_config",
            "design_registration",
        ]
    ]
    assert all(file_hash(n) == h for n, h in guards)
    cfg = json.loads(Path(p["network_config"]).read_text())
    leg_columns = [
        "id",
        "marketing_carrier",
        "operating_carrier",
        "origin",
        "dest",
        "departure_utc_seconds",
        "arrival_utc_seconds",
        "cohort_date",
    ]
    legs = pq.read_table(p["legs"], columns=leg_columns).to_pylist()
    products = pq.read_table(p["products"], columns=["id", "leg_ids"]).to_pylist()
    index = BindingIndex(legs, products)
    db = sqlite3.connect(
        Path(p["quote_db"]).resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    db.execute("PRAGMA query_only=ON")
    source_counts = dict(
        db.execute("SELECT quarter,count(*) FROM products GROUP BY quarter")
    )
    physical_sink = Sink(context.directory / "physical_paths.parquet")
    links_sink = Sink(context.directory / "reported_key_links.parquet")
    physical_buffer = []
    links_buffer = []
    cache = {}
    key_counts = Counter()
    path_counts = Counter()
    nkeys = 0
    current = {
        "stage": "prepared_quote_route_binding",
        "reported_keys_completed": 0,
        "unique_physical_paths": 0,
    }
    context.counters = lambda: dict(current)
    try:
        for identity, payload, quarter, departure, network, kind, market in db.execute(
            "SELECT id,payload,quarter,departure,carrier,kind,market FROM products ORDER BY id"
        ):
            values = json.loads(payload)
            if len(values) != 14:
                raise ValueError("saved reported-key schema changed")
            if (values[0], values[3], values[1] + "-" + values[2]) != (
                departure,
                network,
                market,
            ):
                raise ValueError("saved reported-key metadata disagreement")
            signature = tuple(values[:9])
            found = cache.get(signature)
            if found is None:
                result = bind_reported(
                    list(signature),
                    index,
                    cfg["airport_timezones"],
                    cfg["connection_window_minutes"],
                    "ORD",
                )
                jid = len(cache) + 1
                found = (jid, result["status"], result["n_segments"])
                cache[signature] = found
                result.update(physical_path_id=jid, quarter=quarter)
                # Encode variable nested candidate evidence as exact JSON strings,
                # avoiding data-dependent Arrow null/list inference.
                for name in [
                    "leg_candidates",
                    "departure_candidate_counts",
                    "route_day_candidate_counts",
                    "dated_product_ids",
                ]:
                    result[name] = json.dumps(result[name], separators=(",", ":"))
                result["reason"] = "" if result["reason"] is None else result["reason"]
                result["wait_minutes"] = (
                    float("nan")
                    if result["wait_minutes"] is None
                    else float(result["wait_minutes"])
                )
                physical_buffer.append(result)
                path_counts[(quarter, result["status"], result["n_segments"])] += 1
                if len(physical_buffer) >= p["batch_size"]:
                    physical_sink.write(physical_buffer)
                    physical_buffer.clear()
            jid, status, nsegments = found
            links_buffer.append(
                {
                    "reported_key_id": identity,
                    "physical_path_id": jid,
                    "quarter": quarter,
                }
            )
            key_counts[(quarter, status, nsegments)] += 1
            nkeys += 1
            if len(links_buffer) >= p["batch_size"]:
                links_sink.write(links_buffer)
                links_buffer.clear()
                current.update(
                    reported_keys_completed=nkeys, unique_physical_paths=len(cache)
                )
        physical_sink.write(physical_buffer)
        links_sink.write(links_buffer)
    finally:
        db.close()
        physical_sink.close()
        links_sink.close()
    assert nkeys == sum(source_counts.values())
    coverage = [
        {
            "quarter": q,
            "status": s,
            "segments": n,
            "reported_keys": count,
            "unique_physical_paths": path_counts[(q, s, n)],
        }
        for (q, s, n), count in sorted(key_counts.items())
    ]
    summary = {
        "source_reported_key_counts": source_counts,
        "reported_keys_checked": nkeys,
        "unique_physical_paths": len(cache),
        "dated_legs": len(legs),
        "dated_products": len(products),
        "coverage": coverage,
        "connection_window_minutes": cfg["connection_window_minutes"],
        "timestamp_tolerance_seconds": 0,
        "counts_are": "reported fare/cabin keys and distinct reported physical signatures, not purchases/passengers or full quote rows",
        "scope": "Exact compatibility with existing observed-service dated products under declared model window; not airline legal MCT, booking availability, ticketing identity, hidden stock or calibration.",
        "new_full_quote_scans": 0,
        "new_parameter_fits": 0,
        "new_simulation_cohorts": 0,
        "scientific_budget": None,
        "V1_V5_V6_acceptance": None,
    }
    atomic_json(context.directory / "binding_summary.json", summary)
    assert all(file_hash(n) == h for n, h in guards)
    return {
        "status": "completed",
        **summary,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in [
                "physical_paths.parquet",
                "reported_key_links.parquet",
                "binding_summary.json",
            ]
        },
    }
