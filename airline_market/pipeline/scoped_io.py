"""Bounded local-source I/O for the reduced 2022 study (F8/V1)."""

from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import os
import threading
import zipfile


def atomic_json(path, value):
    """Publish ETL evidence only after the complete record is serialized."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256(path):
    """Hash a source or artifact without loading it into RAM."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def source_stream(path):
    """Open the one data CSV, retaining ZIP documentation as source metadata."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                n
                for n in archive.namelist()
                if n.lower().endswith(".csv")
                and Path(n).name.lower() != "documentation.csv"
            ]
            if len(names) != 1:
                raise ValueError(f"expected one data CSV in {path}; found {names}")
            with archive.open(names[0]) as stream:
                yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def batches(path, columns=None, block_size=8 * 1024**2):
    """Yield actual CSV records in bounded, single-thread Arrow batches."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    with source_stream(path) as stream:
        header = next(csv.reader([stream.readline().decode("utf-8-sig").strip()]))
        stream.seek(0)
        selected = list(columns or [name for name in header if name])
        missing = set(selected) - set(header)
        if missing:
            raise ValueError(f"missing source fields in {path}: {sorted(missing)}")
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=block_size, use_threads=False),
            convert_options=pacsv.ConvertOptions(
                include_columns=selected,
                column_types={name: pa.string() for name in selected},
            ),
        )
        yield from reader


class ParquetSink:
    """Write scoped batches without collecting a full panel in memory."""

    def __init__(self, path):
        """Bind one output; schema is set by the first nonempty batch."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.rows = 0

    def write(self, frame):
        """Append a frame/table using a stable schema and Zstandard compression."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not len(frame):
            return
        table = (
            frame
            if isinstance(frame, pa.Table)
            else pa.Table.from_pandas(frame, preserve_index=False)
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        elif table.schema != self.writer.schema:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += table.num_rows

    def close(self):
        """Finalize the Parquet footer and report actual rows and bytes."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        return {
            "path": str(self.path),
            "rows": self.rows,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
