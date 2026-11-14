"""Deterministic hub/feeder networks with asymmetric product coverage."""

from itertools import product
from ..core.hashing import jsonable
from ..spec import load_spec, compile_market
from ..spec.library_builders import toy_hub
from ..spec.information import add_derived_arms


def hub_feeder_family(params=None):
    """
    Generate a deterministic asymmetric shared-leg hub/feeder network (model §§1–3,
    calibration assumptions).
    """
    p = dict(params or {})
    n = int(p.get("n_spokes", 2))
    T = int(p.get("periods", 6))
    if n < 1 or T < 2:
        raise ValueError("positive spokes and at least two periods required")
    d = jsonable(toy_hub())
    d["name"] = "hub_feeder_" + str(n)
    d["horizon"] = {"n_periods": T}
    d["airports"] = [
        {"code": x}
        for x in ["H"]
        + ["O" + str(k) for k in range(n)]
        + ["D" + str(k) for k in range(n)]
    ]
    legs = []
    products = []
    markets = []
    rules = {}
    base = {"A": {}, "B": {}}
    targets = {}
    capacity = int(p.get("capacity", 3))
    spread = p.get("capacity_support", [1, 3])
    bottleneck = p.get("bottleneck", "rival")
    for k in range(n):
        for suffix, o, z in [("OH", "O" + str(k), "H"), ("HD", "H", "D" + str(k))]:
            mid = f"{suffix}{k}"
            markets.append(
                {"id": mid, "origin": o, "dest": z, "in_segment": suffix == "OH"}
            )
            for carrier in ("A", "B"):
                leg = f"{carrier}_{mid}"
                support = (
                    spread
                    if carrier == "B"
                    and suffix == "HD"
                    and bottleneck in ("rival", "both")
                    else [
                        capacity
                        if not (
                            carrier == "A"
                            and suffix == "OH"
                            and bottleneck in ("focal", "both")
                        )
                        else 2
                    ]
                )
                legs.append(
                    {
                        "id": leg,
                        "carrier": carrier,
                        "origin": o,
                        "dest": z,
                        "capacity": {
                            "support": support,
                            "probabilities": [1 / len(support)] * len(support),
                        },
                    }
                )
                # A does not sell local outbound feeder traffic.
                if carrier == "A" and suffix == "HD":
                    continue
                ref = 110 if carrier == "A" else 125 if suffix == "OH" else 85
                products.append(
                    {
                        "id": leg,
                        "carrier": carrier,
                        "market": mid,
                        "legs": [leg],
                        "reference_fare": ref,
                        "service_cost": 12,
                        "fare_floor": 50,
                        "fare_cap": 400,
                        "priority": len(products),
                    }
                )
                base[carrier][leg] = ref
                if carrier == "A":
                    targets[leg] = "B_" + mid
                else:
                    rules[leg] = {
                        "base": ref,
                        "scarcity": [{"leg": leg, "threshold": 1, "coefficient": 35}],
                    }
    for k, l in product(range(n), repeat=2):
        mid = f"OD{k}_{l}"
        markets.append(
            {
                "id": mid,
                "origin": "O" + str(k),
                "dest": "D" + str(l),
                "in_segment": True,
            }
        )
        for carrier in ("A", "B"):
            pid = carrier + "_" + mid
            ref = 185 if carrier == "A" else 170
            products.append(
                {
                    "id": pid,
                    "carrier": carrier,
                    "market": mid,
                    "legs": [f"{carrier}_OH{k}", f"{carrier}_HD{l}"],
                    "reference_fare": ref,
                    "service_cost": 25,
                    "fare_floor": 50,
                    "fare_cap": 400,
                    "priority": len(products),
                }
            )
            base[carrier][pid] = ref
            if carrier == "A":
                targets[pid] = "B_" + mid
            else:
                rules[pid] = {"base": ref}
    d.update(
        legs=legs,
        products=products,
        markets=markets,
        capacity_priors=[],
        capacity_process={},
        background=d["background"],
    )
    connect = float(p.get("connecting_share", 0.5))
    weights = [
        connect / (n * n) if m["id"].startswith("OD") else (1 - connect) / (2 * n)
        for m in markets
    ]
    d["demand"] = {
        "mode": "single_request_bernoulli",
        "arrival_probability": {"constant": p.get("arrival_probability", 0.95)},
        "market_weights": {"constant": weights},
        "segment_weights": {"piecewise": [[0, [0.8, 0.2]], [T // 2, [0.3, 0.7]]]},
    }
    for seg in d["segments"]:
        for taste in seg["classes"]:
            taste["market_value"] = {"*": 4.5}
            taste["connection_penalty"] = 0.5
    d["choice"] = {"model": "latent_class_mnl"}
    d["exogenous"] = {
        "states": [{"name": "normal"}],
        "initial": [1.0],
        "transitions": {},
    }
    if p.get("demand_factor", "independent") != "independent":
        d["exogenous"] = {
            "states": [
                {"name": "low", "public": False},
                {"name": "high", "public": False},
            ],
            "initial": [0.5, 0.5],
            "transitions": {},
            "default_transition": [[0.9, 0.1], [0.1, 0.9]],
        }
        d["demand"]["exogenous_multipliers"] = {"low": 0.7, "high": 1.0}
    d["pricing"][0]["base"] = base["A"]
    d["pricing"][0]["fixed_features"][0][0]["params"]["targets"] = targets
    d["pricing"][1]["base"] = base["B"]
    d["controllers"] = [
        {"carrier": "A", "controller": "external"},
        {
            "carrier": "B",
            "controller": p.get("rival_controller", "threshold_linear_rule"),
            "params": {"rules": rules}
            if p.get("rival_controller", "threshold_linear_rule")
            == "threshold_linear_rule"
            else {},
        },
    ]
    d["quotes"] = {"requests": []}
    d["information"] = []
    d["timing"] = {"mode": "leader_follower", "leaders": ["B"]}
    d["objectives"] = [
        {
            "carrier": i,
            "profit_scale": 1000.0,
            "share_weight": p.get("share_weight", 0.0),
        }
        for i in ("A", "B")
    ]
    d["metadata"] = {"generator": {"name": "hub_feeder_family", "params": p}}
    # Independent reference quote routes match the linked query count.
    d["background"] = [{**d["background"][0], "id": "XY" + str(k)} for k in range(n)]
    c = compile_market(load_spec(d))
    spec = add_derived_arms(c, "A", lag=int(p.get("lag_periods", 0)))
    d = jsonable(spec)
    for q in d["quotes"]["requests"]:
        q.update(
            fare_bin=p.get("fare_bin", 0), availability=p.get("availability", "flag")
        )
    return load_spec(d)


def independent_legs(params=None):
    """
    Build the same demand markets with independent rival resources (model §§1–3,
    calibration assumptions).
    """
    d = jsonable(hub_feeder_family(params))
    d["name"] = "independent_resources"
    for product_spec in d["products"]:
        if product_spec["carrier"] != "B":
            continue
        new = []
        for k, old in enumerate(product_spec["legs"]):
            leg = next(x.copy() for x in d["legs"] if x["id"] == old)
            leg["id"] = product_spec["id"] + "_resource" + str(k)
            d["legs"].append(leg)
            new.append(leg["id"])
        product_spec["legs"] = new
    # Rules must follow each product's replacement resource.
    assignment = d["controllers"][1]
    if assignment["controller"] == "threshold_linear_rule":
        for pid, rule in assignment["params"]["rules"].items():
            product_spec = next(x for x in d["products"] if x["id"] == pid)
            for term in rule.get("scarcity", []):
                term["leg"] = product_spec["legs"][0]
    c = compile_market(load_spec(d))
    return add_derived_arms(c, "A")


def grid_network(n_spokes, **params):
    """
    Generate a larger spoke network for scaling checks (model §§1–3, calibration
    assumptions).
    """
    return hub_feeder_family({**params, "n_spokes": n_spokes})
