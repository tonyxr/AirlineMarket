"""A.15/V10: bounded DB1C Market scope extraction from one ZIP representation.

Keep every source column and selected row. Missing purchase windows and fares
remain missing. Sampling expansion is explicit; gross amount and allocated tax
are separate raw measures, never an identified airline net-profit measure.
"""

import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REQUIRED = {
    "Origin",
    "Dest",
    "RpYear",
    "RpMonth",
    "SchFlYear",
    "SchFlMonth",
    "Passengers",
    "MktAmount",
    "MktTax",
    "PurchaseWindowGroup",
    "MktCarrier",
}
DERIVED = (
    "_source_row",
    "_purchase_window_status",
    "_event_partition",
    "_sample_expansion",
)


class TrackedMember(io.RawIOBase):
    """Expose seekable ZIP reads while remembering an EOF CRC validation."""

    def __init__(self, stream, size):
        super().__init__()
        self.stream = stream
        self.size = size
        self.crc_verified = False
        self.returned_bytes = 0
        self.backward_seeks = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.stream.tell()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.returned_bytes += len(data)
        if self.stream.tell() == self.size:
            self.crc_verified = True
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset, whence=0):
        old = self.tell()
        value = self.stream.seek(offset, whence)
        self.backward_seeks += int(value < old)
        if value == self.size:
            self.crc_verified = True
        return value

    def close(self):
        if not self.closed:
            self.stream.close()
        super().close()


def _window_status(column):
    text = pc.utf8_upper(pc.utf8_trim_whitespace(pc.cast(column, pa.string())))
    missing = pc.or_(pc.is_null(text), pc.fill_null(pc.equal(text, ""), False))
    known = pc.is_in(text, value_set=pa.array(["21AP", "2290", "91UP"]))
    return pc.if_else(missing, "missing", pc.if_else(known, "reported_group", "other"))


def _partition(year, month):
    y = pc.cast(year, pa.int64())
    m = pc.cast(month, pa.int64())
    pre = pc.fill_null(
        pc.and_(
            pc.equal(y, 2026), pc.and_(pc.greater_equal(m, 1), pc.less_equal(m, 4))
        ),
        False,
    )
    post = pc.fill_null(
        pc.and_(
            pc.equal(y, 2026), pc.and_(pc.greater_equal(m, 5), pc.less_equal(m, 7))
        ),
        False,
    )
    return pc.if_else(
        pre, "pre_event", pc.if_else(post, "post_event", "outside_event_window")
    )


def scope_market_archive(
    path,
    markets,
    output,
    *,
    expected_reporting_period,
    sample_expansion,
    batch_rows=32768,
    progress=None,
):
    """Stream one Parquet member, scope both endpoints, and retain source-row IDs.

    The event partition uses the scheduled first-coupon month, not the reporting
    month; it is a reported monthly label, not an actual dated-flight identity.
    Archive members containing alternate text formats are left untouched.
    """
    if batch_rows < 1 or sample_expansion <= 0:
        raise ValueError("positive batch size and explicit expansion required")
    market_set = sorted({(str(a), str(b)) for a, b in markets})
    if not market_set:
        raise ValueError("an explicit nonempty O&D scope is required")
    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(
            "use a new output; existing complete/partial files are retained"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "plan_id": "A.15/V10",
        "rows_read": 0,
        "rows_kept": 0,
        "reporting_period_mismatch_rows": 0,
        "scope": market_set,
        "sample_expansion": sample_expansion,
        "batch_rows": batch_rows,
        "national_purchase_window_status": Counter(),
        "scoped_event_partitions": Counter(),
        "observed_passengers": 0.0,
        "missing_passenger_rows": 0,
        "scope_observed_passengers": 0.0,
    }
    scope_keys = pa.array([a + "|" + b for a, b in market_set])
    with zipfile.ZipFile(path) as archive:
        members = [
            x for x in archive.infolist() if x.filename.lower().endswith(".parquet")
        ]
        if len(members) != 1:
            raise ValueError("expected exactly one Market Parquet representation")
        info = members[0]
        with TrackedMember(archive.open(info), info.file_size) as stream:
            parquet = pq.ParquetFile(stream, pre_buffer=False, buffer_size=0)
            schema = parquet.schema_arrow
            if REQUIRED - set(schema.names):
                raise ValueError(
                    "missing DB1C fields: " + str(sorted(REQUIRED - set(schema.names)))
                )
            if set(DERIVED) & set(schema.names):
                raise ValueError("reserved provenance field already present")
            dest_schema = (
                schema.append(pa.field("_source_row", pa.int64()))
                .append(pa.field("_purchase_window_status", pa.string()))
                .append(pa.field("_event_partition", pa.string()))
                .append(pa.field("_sample_expansion", pa.float64()))
            )
            with pq.ParquetWriter(partial, dest_schema, compression="zstd") as writer:
                for batch in parquet.iter_batches(
                    batch_size=batch_rows, use_threads=False
                ):
                    table = pa.Table.from_batches([batch])
                    n = table.num_rows
                    status = _window_status(table["PurchaseWindowGroup"])
                    for row in pc.value_counts(status).to_pylist():
                        audit["national_purchase_window_status"][row["values"]] += row[
                            "counts"
                        ]
                    pax = pc.cast(table["Passengers"], pa.float64())
                    total = pc.sum(pax).as_py()
                    audit["observed_passengers"] += total if total is not None else 0.0
                    audit["missing_passenger_rows"] += pax.null_count
                    year, month = expected_reporting_period
                    match = pc.fill_null(
                        pc.and_(
                            pc.equal(table["RpYear"], year),
                            pc.equal(table["RpMonth"], month),
                        ),
                        False,
                    )
                    audit["reporting_period_mismatch_rows"] += n - int(
                        pc.sum(pc.cast(match, pa.int64())).as_py() or 0
                    )
                    keys = pc.binary_join_element_wise(
                        pc.cast(table["Origin"], pa.string()),
                        pc.cast(table["Dest"], pa.string()),
                        "|",
                    )
                    mask = pc.is_in(keys, value_set=scope_keys)
                    offset = audit["rows_read"]
                    table = table.append_column(
                        "_source_row",
                        pa.array(np.arange(offset, offset + n, dtype=np.int64)),
                    )
                    table = table.append_column("_purchase_window_status", status)
                    table = table.append_column(
                        "_event_partition",
                        _partition(table["SchFlYear"], table["SchFlMonth"]),
                    )
                    table = table.append_column(
                        "_sample_expansion", pa.array(np.full(n, sample_expansion))
                    )
                    selected = table.filter(mask)
                    writer.write_table(selected)
                    for row in pc.value_counts(
                        selected["_event_partition"]
                    ).to_pylist():
                        audit["scoped_event_partitions"][row["values"]] += row["counts"]
                    total = pc.sum(
                        pc.cast(selected["Passengers"], pa.float64())
                    ).as_py()
                    audit["scope_observed_passengers"] += (
                        total if total is not None else 0.0
                    )
                    audit["rows_read"] += n
                    audit["rows_kept"] += selected.num_rows
                    if progress is not None:
                        progress(audit["rows_read"], audit["rows_kept"])
            if not stream.crc_verified:
                stream.seek(info.file_size - 1)
                stream.read(1)
            if audit["rows_read"] != parquet.metadata.num_rows:
                raise ValueError("stream row count differs from Parquet footer")
            audit.update(
                member=info.filename,
                member_crc_verified=stream.crc_verified,
                source_columns=schema.names,
                footer_rows=parquet.metadata.num_rows,
                parquet_row_groups=parquet.metadata.num_row_groups,
                arrow_bytes_returned=stream.returned_bytes,
                backward_seeks=stream.backward_seeks,
            )
    partial.replace(output)
    audit.update(
        status="scoped_stream_complete",
        output=str(output),
        full_V10_pass=False,
        inference_scope="Reported monthly O&D purchases; no fit, actual-flight identity, availability, legal itinerary or net-profit identification",
    )
    return audit
