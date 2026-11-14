"""Observed offer-day exposure and display forecasts, not hidden-stock inference."""

import csv
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, skew

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry

CATEGORIES = [str(i) for i in range(11)] + ["other_finite", "missing"]
IDENTITY = [
    "flightDate",
    "startingAirport",
    "destinationAirport",
    "path_network",
    "segmentsDepartureTimeEpochSeconds",
    "segmentsArrivalTimeEpochSeconds",
    "segmentsDepartureAirportCode",
    "segmentsArrivalAirportCode",
    "segmentsAirlineCode",
    "segmentsCabinCode",
    "fareBasisCode",
]
FLAGS = ["isNonStop", "isBasicEconomy", "isRefundable"]
COLUMNS = [
    *IDENTITY,
    *FLAGS,
    "searchDate",
    "quarter",
    "split",
    "days_before_departure",
    "model_path_eligible",
    "left_truncated_by_collection_start",
    "totalFare",
    "seatsRemaining",
]
SEAT_COLUMNS = ["s" + str(i) for i in range(13)]
BIN_SQL = "CASE WHEN day<=2 THEN '0-2' WHEN day<=6 THEN '3-6' WHEN day<=13 THEN '7-13' WHEN day<=20 THEN '14-20' WHEN day<=29 THEN '21-29' WHEN day<=44 THEN '30-44' ELSE '45-59' END"


def text(value):
    return (
        ""
        if value is None or (isinstance(value, float) and not math.isfinite(value))
        else str(value)
    )


def seat_category(value):
    if value is None or not math.isfinite(float(value)):
        return 12
    x = float(value)
    return int(x) if x.is_integer() and 0 <= x <= 10 else 11


def reported_key(row):
    values = [text(row[n]) for n in IDENTITY]
    if any(not x or x.lower() == "nan" for x in values):
        return None
    return canonical_json(values + [text(row[n]).lower() for n in FLAGS])


class MedianCent:
    def __init__(self):
        self.hist = Counter()

    def step(self, cents, count):
        self.hist[int(cents)] += int(count)

    def finalize(self):
        n = sum(self.hist.values())
        positions = [(n - 1) // 2, n // 2]
        found, used = [], 0
        for cents, weight in sorted(self.hist.items()):
            for rank in positions:
                if used <= rank < used + weight:
                    found.append(cents)
            used += weight
        return sum(found) / 2 if n else None


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA journal_mode=TRUNCATE")
    db.executescript(
        """
        CREATE TABLE products(id INTEGER PRIMARY KEY,payload TEXT UNIQUE NOT NULL,
                              quarter INTEGER,departure TEXT,carrier TEXT,kind TEXT,market TEXT);
        CREATE TABLE days(id INTEGER,day INTEGER,n INTEGER,"""
        + ",".join(x + " INTEGER" for x in SEAT_COLUMNS)
        + ",PRIMARY KEY(id,day));"
        "CREATE TABLE fares(id INTEGER,day INTEGER,cents INTEGER,n INTEGER,PRIMARY KEY(id,day,cents));"
    )
    db.create_aggregate("median_cent", 2, MedianCent)
    return db


def ingest(db, path, quarter, p, progress):
    """Project existing Parquet in bounded batches; retain every exclusion count."""
    import pandas as pd
    import pyarrow.parquet as pq

    counters, raw_seats = Counter(), Counter()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=p["batch_size"], columns=COLUMNS
    ):
        frame = batch.to_pandas()
        counters["scanned"] += len(frame)
        expected_split = "calibration" if quarter == 2 else "validation"
        if not ((frame.quarter == quarter) & (frame.split == expected_split)).all():
            raise ValueError("quote quarter/split contamination")
        frame = frame[frame.model_path_eligible].copy()
        counters["core_eligible"] += len(frame)
        if len(frame):
            departures = pd.to_datetime(
                frame.flightDate, format="%Y-%m-%d", errors="raise"
            )
            searches = pd.to_datetime(
                frame.searchDate, format="%Y-%m-%d", errors="raise"
            )
            if not (
                (departures.dt.year == 2022) & (departures.dt.quarter == quarter)
            ).all():
                raise ValueError("departure calendar is outside its declared quarter")
            days = (departures - searches).dt.days
            if (
                not (days.to_numpy() == frame.days_before_departure.to_numpy()).all()
                or (days < 0).any()
            ):
                raise ValueError("source quote date/day mismatch")
            nominal = departures - pd.Timedelta(days=60) < pd.Timestamp(
                p["collection_start"]
            )
            if not (
                nominal.to_numpy()
                == frame.left_truncated_by_collection_start.to_numpy()
            ).all():
                raise ValueError("source left-truncation flag mismatch")
        in_window = frame.days_before_departure.between(0, 59)
        counters["outside_0_59"] += int((~in_window).sum())
        counters["nominal_left_truncated_in_window"] += int(
            (in_window & frame.left_truncated_by_collection_start).sum()
        )
        frame = frame[in_window & ~frame.left_truncated_by_collection_start]
        products, daily, fares = (
            {},
            defaultdict(lambda: np.zeros(14, dtype=np.int64)),
            Counter(),
        )
        for row in frame.to_dict("records"):
            key = reported_key(row)
            if key is None:
                counters["missing_reported_identity"] += 1
                continue
            day = int(row["days_before_departure"])
            flag = text(row["isNonStop"]).lower()
            kind = (
                "nonstop_hub"
                if flag == "true"
                else "connecting"
                if flag == "false"
                else "unknown_reported"
            )
            products[key] = (
                quarter,
                row["flightDate"],
                row["path_network"],
                kind,
                row["startingAirport"] + "-" + row["destinationAirport"],
            )
            daily[(key, day)][0] += 1
            daily[(key, day)][1 + seat_category(row["seatsRemaining"])] += 1
            raw_seats[text(row["seatsRemaining"])] += 1
            fare = row["totalFare"]
            if fare is not None and math.isfinite(float(fare)) and fare > 0:
                fares[(key, day, int(np.rint(fare * 100)))] += 1
            else:
                counters["invalid_fare_retained_for_display"] += 1
            counters["selected_offer_rows"] += 1
        db.executemany(
            "INSERT OR IGNORE INTO products(payload,quarter,departure,carrier,kind,market) VALUES(?,?,?,?,?,?)",
            [(key, *meta) for key, meta in products.items()],
        )
        ids = {}
        keys = list(products)
        for start in range(0, len(keys), 500):
            subset = keys[start : start + 500]
            ids.update(
                {
                    key: identity
                    for identity, key in db.execute(
                        "SELECT id,payload FROM products WHERE payload IN ("
                        + ",".join("?" for _ in subset)
                        + ")",
                        subset,
                    )
                }
            )
        columns = ["n", *SEAT_COLUMNS]
        sql = (
            "INSERT INTO days VALUES("
            + ",".join("?" for _ in range(16))
            + ") ON CONFLICT(id,day) DO UPDATE SET "
            + ",".join(x + "=" + x + "+excluded." + x for x in columns)
        )
        db.executemany(
            sql,
            [
                (ids[key], day, *map(int, counts))
                for (key, day), counts in daily.items()
            ],
        )
        db.executemany(
            "INSERT INTO fares VALUES(?,?,?,?) ON CONFLICT(id,day,cents) DO UPDATE SET n=n+excluded.n",
            [
                (ids[key], day, cents, count)
                for (key, day, cents), count in fares.items()
            ],
        )
        db.commit()
        progress(
            {"stage": "ingest_Q" + str(quarter), "quarter": quarter, **dict(counters)}
        )
    if counters["scanned"] != p["expected_rows"][str(quarter)]:
        raise ValueError("processed quote source row count changed")
    for name in [
        "core_eligible",
        "outside_0_59",
        "nominal_left_truncated_in_window",
        "missing_reported_identity",
        "invalid_fare_retained_for_display",
        "selected_offer_rows",
    ]:
        counters[name] += 0
    return {"counts": dict(counters), "selected_raw_display_counts": dict(raw_seats)}


def groups(db, quarter):
    sql = (
        "SELECT departure,carrier,kind,"
        + BIN_SQL
        + ",COUNT(*),SUM(n),"
        + ",".join("SUM(" + x + "*1.0/n)" for x in SEAT_COLUMNS)
        + " FROM days JOIN products USING(id) WHERE quarter=? GROUP BY departure,carrier,kind,"
        + BIN_SQL
    )
    return list(db.execute(sql, (quarter,)))


def fit_display(rows, alpha=0.5):
    counts = defaultdict(lambda: np.zeros(13))
    for row in rows:
        counts[canonical_json(row[1:4])] += np.asarray(row[6:])
    if not counts:
        raise ValueError("no observed nominal-complete Q2 key-days to fit")
    pooled = sum(counts.values(), np.zeros(13))
    pool = (pooled + alpha) / (pooled.sum() + 13 * alpha)
    kappa = 13 * alpha
    return {
        "target": "Reported display categories only, NOT inventory/availability",
        "fit_quarter": 2,
        "categories": CATEGORIES,
        "half_count": alpha,
        "group_prior_strength": kappa,
        "pooled_counts": pooled.tolist(),
        "pooled": pool.tolist(),
        "groups": {
            key: {
                "weighted_counts": x.tolist(),
                "probability": ((x + kappa * pool) / (x.sum() + kappa)).tolist(),
            }
            for key, x in sorted(counts.items())
        },
    }


def cluster_summary(sums, counts, namespace, B=2000):
    sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
    if len(sums) == 0 or counts.sum() <= 0:
        return {
            "mean": None,
            "clusters": 0,
            "n_units": 0,
            "normal95": None,
            "bootstrap95": None,
        }
    mean = float(sums.sum() / counts.sum())
    G = len(sums)
    if G < 2:
        return {
            "mean": mean,
            "clusters": G,
            "n_units": int(counts.sum()),
            "normal95": None,
            "bootstrap95": None,
        }
    influence = sums - mean * counts
    se = float(np.sqrt(G / (G - 1) * np.sum(influence**2)) / counts.sum())
    shape = float(skew(influence, bias=False)) if G > 2 and se else None
    intervals = {}
    methods = ["percentile"] + (["BCa"] if shape is not None and abs(shape) > 1 else [])
    for method in methods:
        if not se:
            intervals[method] = [mean, mean]
        else:
            boot = bootstrap(
                (sums, counts),
                lambda a, b, axis: np.sum(a, axis=axis) / np.sum(b, axis=axis),
                paired=True,
                n_resamples=B,
                batch=100,
                method=method,
                rng=StreamRegistry(
                    773, "boot/quote-exposure/" + namespace + "/" + method
                ).generator(),
            )
            intervals[method] = [
                float(boot.confidence_interval.low),
                float(boot.confidence_interval.high),
            ]
    return {
        "mean": mean,
        "clusters": G,
        "n_units": int(counts.sum()),
        "cluster_SE": se,
        "normal95": [mean - norm.ppf(0.975) * se, mean + norm.ppf(0.975) * se],
        "bootstrap95": intervals,
        "cluster_influence_skewness": shape,
        "zero_sample_SD_flag": se == 0,
        "interpretation": "Conditional departure-date clusters, not independent offers or model-parameter uncertainty/equivalence.",
    }


def score_display(rows, model):
    pool = np.asarray(model["pooled"])
    dates = defaultdict(lambda: np.zeros(5))
    details = []
    for row in rows:
        date, carrier, kind, days_bin, n_days, n_offers = row[:6]
        counts = np.asarray(row[6:])
        key = canonical_json((carrier, kind, days_bin))
        forecast = np.asarray(
            model["groups"].get(key, {"probability": pool})["probability"]
        )
        if not (forecast > 0).all() or abs(forecast.sum() - 1) > 1e-12:
            raise ValueError("invalid frozen display forecast")
        ll, base_ll = float(counts @ np.log(forecast)), float(counts @ np.log(pool))
        brier = float(n_days * (1 + forecast @ forecast) - 2 * counts @ forecast)
        base_brier = float(n_days * (1 + pool @ pool) - 2 * counts @ pool)
        dates[date] += [ll, base_ll, brier, base_brier, n_days]
        details.append(
            {
                "departure": date,
                "carrier": carrier,
                "market_type": kind,
                "days_bin": days_bin,
                "observed_keydays": n_days,
                "raw_offers": n_offers,
                "weighted_categories": counts.tolist(),
                "forecast": forecast.tolist(),
                "log_score_sum": ll,
                "baseline_log_score_sum": base_ll,
            }
        )
    values = np.asarray(list(dates.values()))
    if not len(values):
        raise ValueError("no held-out observed key-days")
    return {
        "log_score_difference": cluster_summary(
            values[:, 0] - values[:, 1], values[:, 4], "display-LL"
        ),
        "brier_difference": cluster_summary(
            values[:, 2] - values[:, 3], values[:, 4], "display-Brier"
        ),
        "conditional_mean_log_score": float(values[:, 0].sum() / values[:, 4].sum()),
        "pooled_mean_log_score": float(values[:, 1].sum() / values[:, 4].sum()),
        "conditional_mean_Brier": float(values[:, 2].sum() / values[:, 4].sum()),
        "pooled_mean_Brier": float(values[:, 3].sum() / values[:, 4].sum()),
        "departure_dates": len(dates),
        "observed_keydays": int(values[:, 4].sum()),
        "units": "Equal observed reported-product day; display labels only, no latent inventory labels.",
    }, details


def fare_pairs(db, quarter, output):
    """Compare both observed windows within the same exact reported offer key."""
    sql = """SELECT id,departure,carrier,kind,market,
             AVG(CASE WHEN day BETWEEN 45 AND 59 THEN median_cents/100.0 END),
             AVG(CASE WHEN day BETWEEN 0 AND 13 THEN median_cents/100.0 END),
             SUM(day BETWEEN 45 AND 59),SUM(day BETWEEN 0 AND 13)
             FROM daily_fares JOIN products USING(id) WHERE quarter=? GROUP BY id"""
    dates, coverage = defaultdict(lambda: np.zeros(2)), Counter()
    with output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "reported_product_id",
                "departure",
                "carrier",
                "market_type",
                "market",
                "early_mean_daily_median_fare",
                "late_mean_daily_median_fare",
                "early_days",
                "late_days",
                "difference_USD",
            ]
        )
        for row in db.execute(sql, (quarter,)):
            coverage["keys_with_any_valid_fare"] += 1
            if row[5] is None or row[6] is None:
                coverage["unpaired_window_keys"] += 1
                continue
            difference = row[6] - row[5]
            dates[row[1]] += [difference, 1]
            coverage["matched_keys"] += 1
            writer.writerow([*row, difference])
    a = np.asarray(list(dates.values()))
    result = (
        cluster_summary(a[:, 0], a[:, 1], "fare-Q" + str(quarter))
        if len(a)
        else cluster_summary([], [], "empty")
    )
    return {
        **result,
        "coverage": dict(coverage),
        "contrast": "Late0–13 minus early45–59 mean observed-day median all-in fare within exact reported fare-product key.",
        "limitation": "Conditional observed persistence, no missing-day zero fill, no purchase/causal/legally verified itinerary identity claim.",
    }


def run(cell, context):
    p = cell["parameters"]
    for path, h in p["input_hashes"].items():
        if file_hash(Path(path)) != h:
            raise ValueError("quote exposure frozen input changed: " + path)
    receipt = json.loads(Path(p["source_receipt"]).read_text())
    for path in p["quarter_files"].values():
        if receipt["artifact_hashes"][path] != p["input_hashes"][path]:
            raise ValueError("processed quote receipt/input hash mismatch")
    scratch = context.directory / "sqltmp"
    scratch.mkdir()
    os.environ["SQLITE_TMPDIR"] = str(scratch.resolve())
    dbpath = context.directory / "observed_exposure.sqlite"
    if dbpath.exists():
        raise ValueError("new quote-exposure output required; no duplicate processing")
    db = connect(dbpath)
    current = {"stage": "Q2_setup"}
    context.counters = lambda: dict(current)

    def progress(row):
        current.clear()
        current.update(row)

    try:
        q2 = ingest(db, p["quarter_files"]["2"], 2, p, progress)
        atomic_json(context.directory / "Q2_ingestion.json", q2)
        model = fit_display(groups(db, 2), p["prior_half_count"])
        modelpath = context.directory / "frozen_Q2_display_model.json"
        atomic_json(modelpath, model)
        modelhash = file_hash(modelpath)
        atomic_json(
            context.directory / "Q2_model_freeze.json",
            {"model_sha256": modelhash, "Q3_rows_processed_yet": 0, "fit_quarter": 2},
        )
        q3 = ingest(db, p["quarter_files"]["3"], 3, p, progress)
        atomic_json(context.directory / "Q3_ingestion.json", q3)
        if file_hash(modelpath) != modelhash:
            raise ValueError("Q2 model changed during held-out evaluation")
        model = json.loads(modelpath.read_text())
        progress({"stage": "observed_day_summaries"})
        display, details = score_display(groups(db, 3), model)
        atomic_json(context.directory / "Q3_display_groups.json", details)
        atomic_json(context.directory / "Q3_display_scores.json", display)
        db.execute(
            "CREATE TABLE daily_fares AS SELECT id,day,median_cent(cents,n) AS median_cents,SUM(n) AS valid_offers FROM fares GROUP BY id,day"
        )
        db.execute("CREATE INDEX daily_fare_key ON daily_fares(id,day)")
        db.commit()
        paired = {
            str(q): fare_pairs(db, q, context.directory / f"Q{q}_matched_fares.csv")
            for q in [2, 3]
        }
        atomic_json(context.directory / "matched_fare_summary.json", paired)
        coverage = list(
            db.execute(
                "SELECT quarter,observed_days,COUNT(*) FROM (SELECT quarter,id,COUNT(*) AS observed_days FROM products JOIN days USING(id) GROUP BY id) GROUP BY quarter,observed_days"
            )
        )
        if any(x[1] > 60 for x in coverage):
            raise AssertionError(
                "reported product has more than60observed nominalwindow days"
            )
        with (context.directory / "observed_exposure_coverage.csv").open(
            "x", newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(["quarter", "observed_days_in_0_59", "reported_product_keys"])
            w.writerows(coverage)
    finally:
        db.close()
    for path, h in p["input_hashes"].items():
        if file_hash(Path(path)) != h:
            raise ValueError("source changed during quote exposure analysis")
    names = [
        "Q2_ingestion.json",
        "Q3_ingestion.json",
        "frozen_Q2_display_model.json",
        "Q2_model_freeze.json",
        "Q3_display_groups.json",
        "Q3_display_scores.json",
        "Q2_matched_fares.csv",
        "Q3_matched_fares.csv",
        "matched_fare_summary.json",
        "observed_exposure_coverage.csv",
        "observed_exposure.sqlite",
    ]
    result = {
        "status": "completed",
        "schema": "airline-observed-quote-exposure/1",
        "Q2": q2,
        "Q3": q3,
        "Q3_display": display,
        "matched_fares": paired,
        "Q2_model_sha256": modelhash,
        "artifact_hashes": {n: file_hash(context.directory / n) for n in names},
        "scope": "Observed reported-product-day exposure/display forecasting only; missing search days unobserved, 10distinctfrom9, no hidden inventory/availability/CEM/Q3fit/netprofit or V5/V6pass.",
        "full_V5_pass": None,
        "full_V6_pass": None,
        "scientific_calibration_budget": None,
        "test_seeds_opened": False,
    }
    atomic_json(context.directory / "quote_exposure_summary.json", result)
    return {
        **result,
        "summary_sha256": file_hash(context.directory / "quote_exposure_summary.json"),
    }
