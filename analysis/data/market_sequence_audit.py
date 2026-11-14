"""Measure whether DB1B Market sequences identify the declared ORD products."""

from pathlib import Path
import json
import zipfile
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc

AIRPORTS = [
    "ATL",
    "BOS",
    "CLT",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "IAD",
    "JFK",
    "LAX",
    "LGA",
    "MIA",
    "OAK",
    "ORD",
    "PHL",
    "SFO",
]
FIELDS = [
    "Origin",
    "Dest",
    "AirportGroup",
    "MktCoupons",
    "TkCarrierGroup",
    "OpCarrierGroup",
    "TkCarrier",
    "OpCarrier",
    "Passengers",
]


def audit(path):
    """Inspect all qualifying one-carrier local/two-leg ORD records, not a sample."""
    counts = dict(
        rows_scanned=0,
        rows_in_airport_pool=0,
        pool_AA_UA_shortpath_rows=0,
        pool_AA_UA_malformed_sequence_rows=0,
        ord_local_or_one_connection_rows=0,
        marketed_AA_UA_rows=0,
        malformed_airport_sequence_rows=0,
        malformed_ticket_sequence_rows=0,
        malformed_operator_sequence_rows=0,
        aggregate_operator_99_rows=0,
        marketed_AA_UA_passengers=0.0,
    )
    examples = []
    with zipfile.ZipFile(path) as archive:
        member = next(
            n
            for n in archive.namelist()
            if n.lower().endswith(".csv") and "documentation" not in n.lower()
        )
        with archive.open(member) as stream:
            reader = csv.open_csv(
                stream,
                read_options=csv.ReadOptions(block_size=8 * 1024**2, use_threads=False),
                convert_options=csv.ConvertOptions(
                    include_columns=FIELDS,
                    column_types={h: pa.string() for h in FIELDS},
                ),
            )
            for batch in reader:
                counts["rows_scanned"] += batch.num_rows
                table = pa.Table.from_batches([batch])
                table = table.filter(
                    pc.and_(
                        pc.is_in(table["Origin"], value_set=pa.array(AIRPORTS)),
                        pc.is_in(table["Dest"], value_set=pa.array(AIRPORTS)),
                    )
                )
                counts["rows_in_airport_pool"] += table.num_rows
                for row in table.to_pylist():
                    n = int(float(row["MktCoupons"]))
                    if n not in (1, 2):
                        continue
                    tk = (row["TkCarrierGroup"] or "").split(":")
                    ap = (row["AirportGroup"] or "").split(":")
                    op = (row["OpCarrierGroup"] or "").split(":")
                    touches_ord = "ORD" in ap or "ORD" in (row["Origin"], row["Dest"])
                    counts["ord_local_or_one_connection_rows"] += touches_ord
                    candidate = row["TkCarrier"] in ("AA", "UA") or set(tk) in (
                        {"AA"},
                        {"UA"},
                    )
                    if not candidate:
                        continue
                    bad_ap = (
                        len(ap) != n + 1
                        or ap[0] != row["Origin"]
                        or ap[-1] != row["Dest"]
                    )
                    bad_tk = len(tk) != n or not all(tk)
                    bad_op = len(op) != n or not all(op)
                    counts["pool_AA_UA_shortpath_rows"] += 1
                    counts["pool_AA_UA_malformed_sequence_rows"] += (
                        bad_ap or bad_tk or bad_op
                    )
                    if not touches_ord or set(tk) not in ({"AA"}, {"UA"}):
                        continue
                    counts["marketed_AA_UA_rows"] += 1
                    counts["marketed_AA_UA_passengers"] += float(row["Passengers"])
                    counts["malformed_airport_sequence_rows"] += bad_ap
                    counts["malformed_ticket_sequence_rows"] += bad_tk
                    counts["malformed_operator_sequence_rows"] += bad_op
                    counts["aggregate_operator_99_rows"] += row["OpCarrier"] == "99"
                    if (bad_ap or bad_tk or bad_op) and len(examples) < 5:
                        examples.append(row)
    return {
        "archive": str(path),
        "counts": counts,
        "malformed_examples": examples,
        "scope": "16-airport pool, local/one-connection paths containing ORD, all coupons marketed AA or all marketed UA. Other carrier/path records remain necessary when constructing selected-market traffic denominators.",
        "coupon_supplement_needed_for_sequence_lengths": any(
            counts[k] > 0
            for k in [
                "malformed_airport_sequence_rows",
                "malformed_ticket_sequence_rows",
                "malformed_operator_sequence_rows",
                "pool_AA_UA_malformed_sequence_rows",
            ]
        ),
    }


if __name__ == "__main__":
    results = [
        audit(Path(f"data/raw/bts/db1b/zip/DB1BMarket_2022_{q}.zip")) for q in (2, 3)
    ]
    path = Path("results/data_acquisition/db1b_market_sequence_audit.json")
    path.write_text(
        json.dumps(
            {
                "quarters": results,
                "note": "Sequence integrity is measured, not a claim that quote data identify preferences or that the final hub meets F8 traffic/coverage thresholds.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))
