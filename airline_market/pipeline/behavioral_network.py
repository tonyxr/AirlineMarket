"""Q2-initialized behavioral cohorts over the audited dated physical tables.

This is an uncalibrated initializer. Outside-network ticket purchases have
an explicit aggregate alternative; the utility-zero option is nonpurchase.
All endogenous prices are all-in. Native engine profit remains gross-fare
contribution until an explicitly compatible net-revenue objective is used.
"""

from datetime import datetime
import math
from pathlib import Path

import numpy as np

from ..spec import load_spec


def weighted_median(values, weights):
    """Take a weighted median of target-bin medians, not a claimed pooled median."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    if len(v) == 0 or not np.isfinite(v).all() or w.sum() <= 0:
        raise ValueError("initialization needs finite positive-weight fare targets")
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def initialize_cohort(legs, products, traffic, fare_targets, config, date):
    """Preserve physical resources and initialize behavior from Q2-only targets."""
    if set(legs.cohort_date) != {date} or set(products.cohort_date) != {date}:
        raise ValueError("dated physical tables must contain exactly the requested cohort")
    carriers = config["core_carriers"]
    if not set(products.carrier) <= set(carriers):
        raise ValueError("unexpected core product carrier")
    T = int(config["periods"])
    conversion = float(config["assumed_conversion"])
    if not 0 < conversion < 1:
        raise ValueError("conversion initialization must be strictly between zero and one")
    resources = []
    for row in legs.itertuples(index=False):
        low, high, p = int(row.capacity_low), int(row.capacity_high), float(row.probability_high)
        resources.append({"id": row.id, "carrier": row.marketing_carrier,
                          "origin": row.origin, "dest": row.dest,
                          "dep_time": row.departure_local, "arr_time": row.arrival_local,
                          "distance_mi": float(row.distance_mi),
                          "capacity": {"support": [low, high] if low != high else [low],
                                       "probabilities": [1 - p, p] if low != high else [1.0],
                                       "public": False}})
    offers, markets, fare_methods = [], [], {}
    fare_groups = {(m, c, k): g for (m, c, k), g in fare_targets.groupby(
        ["market", "path_network", "market_type"])}
    fare_prior = weighted_median(fare_targets.median_total_fare, fare_targets.observed_offers)
    floor, cap = config["fare_bound_multipliers"]
    for row in products.sort_values("id").itertuples(index=False):
        kind = "nonstop_hub" if row.n_legs == 1 else "connecting"
        key = row.market, row.carrier, kind
        target = fare_groups.get(key)
        ref = (weighted_median(target.median_total_fare, target.observed_offers)
               if target is not None else fare_prior)
        fare_methods["|".join(key)] = ("Q2 weighted median of bin medians"
            if target is not None else "declared sparse-cell prior: pooled Q2 target-bin median")
        offers.append({"id": row.id, "carrier": row.carrier, "market": row.market,
                       "legs": list(row.leg_ids), "service_cost":
                           config["assumed_cost_per_pax_mile"] * float(row.distance_mi),
                       "fare_floor": floor * ref, "fare_cap": cap * ref,
                       "reference_fare": ref, "priority": len(offers),
                       "attributes": {"duration_h": float(row.duration_h),
                                      "dep_hour": datetime.fromisoformat(row.departure_local).hour
                                          + datetime.fromisoformat(row.departure_local).minute / 60}})
    masses, intensities, outside_rows = {}, {}, []
    for mid, group in products.groupby("market", sort=True):
        o, d = mid.split("-")
        markets.append({"id": mid, "origin": o, "dest": d,
                        "in_segment": set(carriers) <= set(group.carrier)})
        obs = traffic[traffic.market == mid]
        total = float(obs.passengers.sum())
        if total <= 0:
            raise ValueError("dated market has no Q2 all-carrier traffic target: " + mid)
        core = {c: float(obs.loc[obs.model_path_eligible & (obs.path_network == c),
                                "passengers"].sum()) if c in set(group.carrier) else 0.0
                for c in carriers}
        residual = total - sum(core.values())
        if residual < -1e-8:
            raise ValueError("represented traffic exceeds all-carrier denominator")
        masses[mid] = {c: max(config["share_pseudocount"], n) for c, n in core.items()
                       if c in set(group.carrier)}
        masses[mid]["OUTSIDE"] = max(config["share_pseudocount"], residual)
        outside_rows.append({"market": mid, "all_carrier_Q2_passengers": total,
                             "represented_Q2_passengers": sum(core.values()),
                             "outside_network_Q2_passengers": residual,
                             "outside_share_is_nonpurchase": False})
        intensities[mid] = {"constant": total / config["Q2_calendar_days"] / conversion / T}
        proxy_id = "OUTSIDE_" + mid
        resources.append({"id": proxy_id, "carrier": "OUTSIDE", "origin": o, "dest": d,
                          "capacity": {"support": [config["outside_proxy_capacity"]],
                                       "probabilities": [1.0], "public": True}})
        offers.append({"id": proxy_id, "carrier": "OUTSIDE", "market": mid,
                       "legs": [proxy_id], "service_cost": 0.0, "fare_floor": fare_prior,
                       "fare_cap": fare_prior, "reference_fare": fare_prior,
                       "priority": len(offers), "controlled": False})
    segments = []
    for prior in config["segments"]:
        shifts = {}
        for mid in masses:
            total_mass = sum(masses[mid].values())
            local = [p for p in offers if p["market"] == mid]
            for p in local:
                count = sum(q["carrier"] == p["carrier"] for q in local)
                shifts[p["id"]] = (math.log(masses[mid][p["carrier"]] / total_mass / count)
                    + prior["alpha"] * p["reference_fare"]
                    + prior["connection_penalty"] * (len(p["legs"]) - 1))
        segments.append({"id": prior["id"], "classes": [{"weight": 1.0,
            "alpha": prior["alpha"], "carrier_pref": prior["carrier_pref"],
            "market_value": {"*": math.log(conversion / (1 - conversion))},
            "connection_penalty": prior["connection_penalty"], "product_shift": shifts}]})
    pricing, controllers, information, requests = [], [], [], []
    for c in [*carriers, "OUTSIDE"]:
        own = [p for p in offers if p["carrier"] == c]
        active = c != "OUTSIDE"
        pricing.append({"carrier": c, "base": {p["id"]: p["reference_fare"] for p in own},
                        "controllable_features": ([{"name": "intercept"},
                            {"name": "connection_indicator"}, {"name": "scarcity_sum"}]
                            if active else []),
                        "theta_bounds": [[-50, 50], [-50, 50], [0, 30]] if active else [],
                        "theta_grid": [[0, 0, 0]] if active else [[]]})
        controllers.append({"carrier": c, "controller": "fixed",
                            "params": {"theta": [0, 0, 0] if active else []}})
        if not active:
            continue
        own_markets = {p["market"] for p in own}
        rival = [p for p in offers if p["carrier"] in carriers and p["carrier"] != c]
        direct = [p["id"] for p in rival if p["market"] in own_markets]
        extra = [p["id"] for p in rival if p["market"] not in own_markets]
        names = []
        for label, targets in [("direct", direct), ("linked_extra", extra)]:
            if targets:
                name = c + "_" + label
                requests.append({"id": name, "observer": c, "target_products": targets,
                                 "availability": "flag", "lag_periods": 1})
                names.append(name)
        information.extend([{"name": "direct", "observer": c,
                             "requests": [c + "_direct"] if direct else []},
                            {"name": "linked", "observer": c, "requests": names}])
    metadata = {"focal": carriers[0], "calibrated": False, "initialization_split": "Q2 only",
                "scope": "observed-service core physical network plus explicit aggregate outside-network ticket proxy",
                "nonpurchase": "utility-zero alternative, distinct from OUTSIDE product",
                "outside_proxy": {"synthetic_resource": True,
                    "capacity": config["outside_proxy_capacity"],
                    "interpretation": "finite technical proxy, not measured supply or an infinite-capacity theorem"},
                "outside_traffic": outside_rows, "fare_initialization": fare_methods,
                "assumptions": config,
                "identification": "Arrival conversion/profile, two-segment tastes/preferences, costs and outside aggregation are initial assumptions, not CEM estimates. Reference log-share shifts cancel product multiplicity; preference priors can perturb shares. Q2 tickets do not identify nonpurchase.",
                "accounting": "Native engine objective uses all-in gross fares; base/all-in sidecar is a proxy diagnostic, not an optimized empirical net-profit objective.",
                "supply": "Ex-post performed service; no forward cancellation/weather forecast fitted."}
    return load_spec({"schema_version": "airline-market-spec/1", "name": "ORD_" + date,
        "horizon": {"n_periods": T, "period_days": list(range(T, 0, -1)),
                    "departure_date": date, "booking_window_days": T},
        "airports": [{"code": c, "timezone": zone} for c, zone in config["airport_timezones"].items()],
        "carriers": [{"code": c} for c in carriers] + [{"code": "OUTSIDE", "kind": "outside_network_proxy"}],
        "legs": resources, "products": offers, "markets": markets, "segments": segments,
        "demand": {"mode": "poisson", "intensity": intensities,
                   "segment_weights": {"constant": [p["mix"] for p in config["segments"]]}},
        "choice": {}, "exogenous": {"states": [{"name": "normal"}], "initial": [1.0],
                                     "transitions": {}, "default_transition": [[1.0]]},
        "capacity_process": {}, "resolution": {}, "quotes": {"requests": requests},
        "information": information, "pricing": pricing, "controllers": controllers,
        "objectives": [{"carrier": c, "profit_scale": 100000.0} for c in [*carriers, "OUTSIDE"]],
        "timing": {"mode": "simultaneous"}, "metadata": metadata})


def read_q2_inputs(config, date):
    """Read only dated structure and previously prepared Q2 behavioral targets."""
    import pandas as pd

    network, targets = Path(config["dated_network"]), Path(config["Q2_targets"])
    legs = pd.read_parquet(network / "legs.parquet", filters=[("cohort_date", "==", date)])
    products = pd.read_parquet(network / "products.parquet", filters=[("cohort_date", "==", date)])
    traffic = pd.read_csv(targets / "q2_passenger_targets.csv")
    fares = pd.read_csv(targets / "q2_quote_targets_by_market.csv")
    return legs, products, traffic, fares
