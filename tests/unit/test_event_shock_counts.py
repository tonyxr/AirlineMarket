"""A.15/B1/V10: NEW local prepared-index fixtures, no raw/source-bank replay."""

import copy
import hashlib
import importlib.util
import os
import sqlite3
import sys
from fractions import Fraction

import pytest

if os.environ.get("B1_COUNTS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_event_counts", os.environ["B1_COUNTS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.pipeline import event_shock_counts as module


def definition():
    return module.EventCountWindow(
        "NK", (("A", "B"), ("B", "A")), "2026-05-02", "2026-05-04"
    )


def databases(tmp_path, extra=()):
    left = tmp_path / "reported.sqlite"
    right = tmp_path / "service.sqlite"
    with sqlite3.connect(left) as db:
        db.execute(
            "CREATE TABLE rows(sid TEXT PRIMARY KEY,k TEXT,opid INTEGER,op TEXT,o TEXT,d TEXT,c INTEGER,v INTEGER,utc_id TEXT)"
        )
        db.executemany(
            "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?)",
            [
                ("r1", "f1", 10, "NK", "A", "B", 1, 0, None),
                ("r2", "f1", 10, "NK", "A", "B", 1, 0, None),
                ("r3", "f2", 10, "NK", "A", "B", 0, 1, "utc2"),
                ("r4", "f3", 10, "NK", "B", "A", 0, 0, "utc3"),
                ("old", "old", 10, "NK", "A", "B", 1, 0, "old"),
                ("other", "other", 20, "AA", "A", "B", 1, 0, "other"),
                *extra,
            ],
        )
    with sqlite3.connect(right) as db:
        db.execute(
            "CREATE TABLE rows(row_id TEXT PRIMARY KEY,day TEXT,origin TEXT,dest TEXT,op TEXT)"
        )
        db.executemany(
            "INSERT INTO rows VALUES(?,?,?,?,?)",
            [
                ("r1", "2026-05-02", "A", "B", "NK"),
                ("r2", "2026-05-02", "A", "B", "NK"),
                ("r3", "2026-05-02", "A", "B", "NK"),
                ("r4", "2026-05-02", "B", "A", "NK"),
                ("old", "2026-05-01", "A", "B", "NK"),
                ("other", "2026-05-02", "A", "B", "AA"),
            ],
        )
    return left, right


def profile(tmp_path, extra=()):
    left, right = databases(tmp_path, extra)
    joined = module.join_prepared_indexes(left, right, definition())
    return joined, module.profile_joined_rows(joined, definition())


def fraction(row):
    r = row["reported_cancellation_fraction"]
    return None if r is None else Fraction(int(r["numerator"]), int(r["denominator"]))


def test_exact_id_join_deduplicates_marketing_keeps_utc_errors_and_diversions(tmp_path):
    joined, out = profile(tmp_path)
    a, b, c, d = out["rows"]
    assert (
        joined["source_operator_rows"] == 5
        and joined["outside_declared_window_or_routes"] == 1
    )
    assert a["source_rows"] == 3 and a["identified_reported_flights"] == 2
    assert (
        a["known_cancelled"] == 1 and a["known_non_cancelled_including_diversions"] == 1
    )
    assert a["non_cancelled_diversion"] == 1 and a["completed_no_diversion_proxy"] == 0
    assert a["utc_unresolved_rows"] == 2 and fraction(a) == Fraction(1, 2)
    assert fraction(b) == 0 and b["completed_no_diversion_proxy"] == 1
    assert fraction(c) is fraction(d) is None
    assert c["rate_null_reasons"] == ["no_reported_source_rows"]
    assert all(
        r["native_installed_seat_loss_share"] is r["B1_physical_range_check"] is None
        for r in out["rows"]
    )
    assert not out["full_B1_anchoring"] and not out["native_capacity_adopted"]


def test_database_files_are_read_only_and_source_unchanged(tmp_path):
    left, right = databases(tmp_path)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (left, right)]
    module.join_prepared_indexes(left, right, definition())
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (left, right)]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "reported.sqlite",
        "service.sqlite",
    ]


def test_nonexistent_source_is_not_silently_created(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        module.join_prepared_indexes(
            tmp_path / "missing.sqlite", tmp_path / "other.sqlite", definition()
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cancelled", [None, 2, 0])
def test_missing_invalid_and_conflicting_cancellation_stay_unknown(tmp_path, cancelled):
    left, right = databases(tmp_path)
    with sqlite3.connect(left) as db:
        db.execute("UPDATE rows SET c=? WHERE sid=?", (cancelled, "r2"))
    out = module.profile_joined_rows(
        module.join_prepared_indexes(left, right, definition()), definition()
    )
    a = out["rows"][0]
    assert fraction(a) is None and a["unknown_cancellation"] == 1
    assert a["known_cancelled"] == 0
    assert "unknown_or_conflicting_cancellation" in a["rate_null_reasons"]


def test_missing_identity_never_guessed_from_utc(tmp_path):
    left, right = databases(tmp_path)
    with sqlite3.connect(left) as db:
        db.execute("UPDATE rows SET k=NULL WHERE sid='r3'")
    out = module.profile_joined_rows(
        module.join_prepared_indexes(left, right, definition()), definition()
    )
    assert out["rows"][0]["unresolved_identity_rows"] == 1
    assert fraction(out["rows"][0]) is None


def test_unassignable_date_join_retained_and_rates_null_without_nearest_fill(tmp_path):
    joined, out = profile(
        tmp_path, [("orphan", "lost", 10, "NK", "A", "B", 1, 0, "utc")]
    )
    assert joined["unassignable_rows"][0]["source_id"] == "orphan"
    assert out["unassignable_operator_rows"] == 1
    assert all(fraction(r) is None for r in out["rows"])


def test_inconsistent_route_or_operator_join_does_not_certify_rate(tmp_path):
    left, right = databases(tmp_path)
    with sqlite3.connect(right) as db:
        db.execute("UPDATE rows SET origin='X' WHERE row_id='r1'")
    out = module.profile_joined_rows(
        module.join_prepared_indexes(left, right, definition()), definition()
    )
    assert out["rows"][0]["join_inconsistent_rows"] == 1
    assert fraction(out["rows"][0]) is None


def test_identity_crossing_dates_is_retained_and_flagged(tmp_path):
    left, right = databases(tmp_path)
    with sqlite3.connect(right) as db:
        db.execute("UPDATE rows SET day='2026-05-03' WHERE row_id='r2'")
    out = module.profile_joined_rows(
        module.join_prepared_indexes(left, right, definition()), definition()
    )
    assert (
        out["rows"][0]["cross_group_flights"]
        == out["rows"][2]["cross_group_flights"]
        == 1
    )
    assert fraction(out["rows"][0]) is fraction(out["rows"][2]) is None


def test_unknown_diversion_does_not_erase_known_non_cancellation(tmp_path):
    left, right = databases(tmp_path)
    with sqlite3.connect(left) as db:
        db.execute("UPDATE rows SET v=NULL WHERE sid='r3'")
    out = module.profile_joined_rows(
        module.join_prepared_indexes(left, right, definition()), definition()
    )
    assert fraction(out["rows"][0]) == Fraction(1, 2)
    assert out["rows"][0]["non_cancelled_unknown_diversion"] == 1


def test_new_saved_join_review_needs_no_source_reopening(tmp_path, monkeypatch):
    joined, out = profile(tmp_path)
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no old input reopen")),
    )
    assert module.profile_joined_rows(copy.deepcopy(joined), definition()) == out
    changed = copy.deepcopy(joined)
    changed["rows"].append(changed["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        module.profile_joined_rows(changed, definition())


@pytest.mark.parametrize(
    "start,stop",
    [
        ("2026-05-02", "2026-05-02"),
        ("20260502", "2026-05-03"),
        ("2026-05-03", "2026-05-02"),
    ],
)
def test_window_is_explicit_iso_half_open(start, stop):
    with pytest.raises(ValueError):
        module.EventCountWindow("NK", (("A", "B"),), start, stop)


def test_fraction_is_exact_and_partition_tampering_refused(tmp_path):
    left, right = databases(tmp_path)
    with sqlite3.connect(left) as db:
        db.execute("INSERT INTO rows VALUES('third','f4',10,'NK','A','B',0,0,'utc4')")
    with sqlite3.connect(right) as db:
        db.execute("INSERT INTO rows VALUES('third','2026-05-02','A','B','NK')")
    joined = module.join_prepared_indexes(left, right, definition())
    out = module.profile_joined_rows(joined, definition())
    assert out["rows"][0]["reported_cancellation_fraction"] == {
        "numerator": "1",
        "denominator": "3",
    }
    joined["outside_declared_window_or_routes"] += 1
    with pytest.raises(ValueError, match="conserve"):
        module.profile_joined_rows(joined, definition())
