"""Deterministic dated-network assembly; defaults remain explicit assumptions."""

from pathlib import Path
import math
from ..core.errors import DataValidationError
from ..spec import load_spec, compile_market
from ..spec.information import add_derived_arms
from .sources import t100, db1b, ontime
from .demand import booking_curve, alpha_from_elasticity

AIRCRAFT_DEFAULT_SEATS = {"default": 150, "regional": 76}


def minutes(value):
    """
    Convert a scheduled HHMM field to minutes after local midnight (model §§1–3,
    calibration assumptions).
    """
    value = int(float(value))
    return value // 100 * 60 + value % 100


def compile_scope(scope):
    """
    Assemble dated resources, products and explicitly assumed demand/choice inputs
    (model §§1–3, calibration assumptions).
    """
    if scope.builder == "library":
        return load_spec(
            Path(__file__).parents[1] / "spec/library" / scope.params["spec"]
        )
    capacity = t100.load_segments(scope)
    fares = db1b.load_markets(scope)
    schedule = ontime.load_schedule(scope)
    schedule = schedule[
        (schedule.cancelled == 0)
        & (schedule.origin.isin(scope.hubs) | schedule.dest.isin(scope.hubs))
    ].sort_values(["carrier", "origin", "dest", "crs_dep_time", "flight_number"])
    params = scope.params
    points = params.get("periods", [60, 30, 14, 7, 0])
    T = len(points) - 1
    if T < 2:
        raise DataValidationError("at least three collection points required")
    warnings = [
        "Demand conversion, booking curve, outside share, elasticities and service costs are assumed calibration parameters; aggregate data do not identify them separately."
    ]
    legs = []
    dated = []
    spread = int(params.get("capacity_spread", 1))
    month = int(scope.departure_date[5:7])
    for row in schedule.itertuples():
        subset = capacity[
            (capacity.carrier == row.carrier)
            & (capacity.origin == row.origin)
            & (capacity.dest == row.dest)
        ]
        if "month" in subset:
            subset = subset[subset.month == month]
        valid = subset.seats_per_departure.dropna()
        seats = (
            max(1, int(round(valid.mean())))
            if len(valid)
            else int(params.get("default_seats", AIRCRAFT_DEFAULT_SEATS["default"]))
        )
        if not len(valid):
            warnings.append(
                f"assumed default seats for {row.carrier}/{row.origin}/{row.dest}"
            )
        pid = f"{row.carrier}_{row.flight_number}_{row.origin}_{row.dest}"
        if any(x["id"] == pid for x in legs):
            raise DataValidationError("duplicate dated flight identity " + pid)
        support = sorted(set(max(0, seats + x) for x in range(-spread, spread + 1)))
        dep = minutes(row.crs_dep_time)
        arr = minutes(row.crs_arr_time)
        arr += 1440 * (arr < dep)
        distance = (
            float(subset.distance.mean())
            if "distance" in subset and subset.distance.notna().any()
            else float(params.get("default_distance_mi", 500))
        )
        legs.append(
            {
                "id": pid,
                "carrier": row.carrier,
                "origin": row.origin,
                "dest": row.dest,
                "dep_time": f"{scope.departure_date}T{dep // 60:02}:{dep % 60:02}:00",
                "arr_time": None,
                "distance_mi": distance,
                "capacity": {
                    "support": support,
                    "probabilities": [1 / len(support)] * len(support),
                    "public": params.get("capacity_public", False),
                },
            }
        )
        dated.append((pid, row.carrier, row.origin, row.dest, dep, arr, distance))
    if not legs:
        raise DataValidationError("scope has no active dated legs")
    itineraries = [(x[1], x[2], x[3], [x[0]], x[4], x[5], x[6]) for x in dated]
    window = params.get("connection_window_min", [40, 300])
    for a in dated:
        for b in dated:
            if (
                a[1] == b[1]
                and a[3] == b[2]
                and a[3] in scope.hubs
                and a[2] != b[3]
                and window[0] <= b[4] - a[5] <= window[1]
            ):
                itineraries.append(
                    (a[1], a[2], b[3], [a[0], b[0]], a[4], b[5], a[6] + b[6])
                )
    products = []
    markets = {}
    market_passengers = {}
    market_shares = {}
    days = float(params.get("sample_days", 92))
    minimum = params.get("min_market_pax_per_day", 0.0)
    for carrier, o, z, path, dep, arr, dist in itineraries:
        allfare = fares[(fares.origin == o) & (fares.dest == z)]
        daily = float(allfare.passengers.sum() / days)
        if daily < minimum:
            continue
        f = allfare[(allfare.carrier == carrier) & (allfare.coupons == len(path))]
        if f.empty:
            f = allfare[allfare.carrier == carrier]
        if f.empty:
            ref = float(params.get("default_fare", 200))
            floor = ref * 0.4
            cap = ref * 3
            warnings.append(f"assumed fare for {carrier}/{o}/{z}/{len(path)}")
        else:
            ref = float(f.p50.median())
            floor = max(1.0, float(f.p05.min()))
            cap = max(ref, float(f.p99.max()) * 1.5)
        mid = o + "-" + z
        markets[mid] = {"id": mid, "origin": o, "dest": z, "in_segment": False}
        market_passengers[mid] = daily
        share = float(
            allfare[allfare.carrier == carrier].passengers.sum()
            / max(1, allfare.passengers.sum())
        )
        market_shares[(mid, carrier)] = share
        products.append(
            {
                "id": carrier + "_" + str(len(products)),
                "carrier": carrier,
                "market": mid,
                "legs": path,
                "service_cost": params.get("cost_per_pax_mile", 0.04) * dist,
                "fare_floor": floor,
                "fare_cap": cap,
                "reference_fare": ref,
                "priority": len(products),
                "attributes": {"duration_h": (arr - dep) / 60, "dep_hour": dep / 60},
            }
        )
    if not products:
        raise DataValidationError("no products pass the market passenger threshold")
    active_carriers = tuple(
        i for i in scope.carriers if any(p["carrier"] == i for p in products)
    )
    if len(active_carriers) < len(scope.carriers):
        warnings.append(
            "carriers without offered products are excluded from the compiled game"
        )
    legs = [l for l in legs if l["carrier"] in active_carriers]
    for mid, m in markets.items():
        m["in_segment"] = (
            len({p["carrier"] for p in products if p["market"] == mid}) >= 2
        )
    outside = float(params.get("outside_share", 0.3))
    conversion = float(params.get("conversion_rate", 0.6))
    curve = booking_curve(
        points, params.get("booking_shape", 2.0), params.get("booking_scale", 12.0)
    )
    intensity = {
        mid: {
            "piecewise": [
                [t, float(daily / conversion * w)] for t, w in enumerate(curve)
            ]
        }
        for mid, daily in market_passengers.items()
    }
    segments = []
    for label, elasticity in [("leisure", -1.6), ("business", -0.7)]:
        ref = sum(p["reference_fare"] for p in products) / len(products)
        alpha = alpha_from_elasticity(
            params.get(label + "_elasticity", elasticity),
            ref,
            (1 - outside) / len(active_carriers),
        )
        preferences = {
            i: sum(
                math.log(max(0.001, market_shares.get((m, i), 0.001)) / outside)
                for m in markets
            )
            / len(markets)
            + alpha * ref
            for i in active_carriers
        }
        segments.append(
            {
                "id": label,
                "classes": [
                    {
                        "weight": 1.0,
                        "alpha": alpha,
                        "carrier_pref": preferences,
                        "market_value": {"*": 0.0},
                        "connection_penalty": params.get("connection_penalty", 0.5),
                    }
                ],
            }
        )
    pricing = [
        {
            "carrier": i,
            "base": {
                p["id"]: p["reference_fare"] for p in products if p["carrier"] == i
            },
            "controllable_features": [
                {"name": "intercept"},
                {"name": "connection_indicator"},
                {"name": "scarcity_sum"},
            ],
            "theta_bounds": [[-50, 50], [-50, 50], [0, 30]],
            "theta_grid": [[-30, 0, 0], [0, 0, 0], [30, 0, 15]],
        }
        for i in active_carriers
    ]
    focal = params.get("focal", active_carriers[0])
    spec = {
        "schema_version": "airline-market-spec/1",
        "name": scope.name,
        "horizon": {
            "n_periods": T,
            "period_days": points[:-1],
            "departure_date": scope.departure_date,
            "booking_window_days": points[0],
        },
        "airports": [
            {"code": x}
            for x in sorted({l["origin"] for l in legs} | {l["dest"] for l in legs})
        ],
        "carriers": [{"code": x} for x in active_carriers],
        "legs": legs,
        "products": products,
        "markets": list(markets.values()),
        "segments": segments,
        "demand": {
            "mode": "poisson",
            "intensity": intensity,
            "segment_weights": {
                "constant": [
                    1 - params.get("business_share", 0.3),
                    params.get("business_share", 0.3),
                ]
            },
        },
        "choice": {},
        "exogenous": {
            "states": [{"name": "normal"}],
            "initial": [1.0],
            "transitions": {},
        },
        "capacity_process": {},
        "resolution": {},
        "quotes": {"requests": []},
        "information": [],
        "pricing": pricing,
        "controllers": [
            {
                "carrier": i,
                "controller": "external"
                if i == focal
                else params.get("rival_controller", "bid_price_dlp"),
            }
            for i in active_carriers
        ],
        "objectives": [
            {"carrier": i, "profit_scale": params.get("profit_scale", 10000.0)}
            for i in active_carriers
        ],
        "timing": {"mode": "simultaneous"},
        "metadata": {
            "focal": focal,
            "source_hashes": {
                k: f.attrs.get("sources")
                for k, f in [("t100", capacity), ("db1b", fares), ("ontime", schedule)]
            },
            "parameters": params,
            "identification": {
                "carrier_preferences": {
                    "status": "assumed_initialization",
                    "method": "cross-market average of scoped carrier log-share inversions; not an estimated structural preference",
                },
                "market_value": {"status": "assumed_normalization", "value": 0.0},
                "outside_option": {
                    "status": "assumed",
                    "share": outside,
                    "scope": "DB1B market totals are filtered to scope carriers; out-of-scope airlines and nonpurchase are absorbed into the outside option",
                },
                "calibration": "aggregate moments alone do not identify these assumptions",
            },
            "warnings": warnings,
            "leg_aggregation": "none",
        },
    }
    if params.get("leg_aggregation", "none") != "none":
        raise DataValidationError("time-band leg aggregation is not enabled")
    result = load_spec(spec)
    result = add_derived_arms(compile_market(result), focal, lag=1)
    return result
