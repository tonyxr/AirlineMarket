"""Small specified networks, including the exact legacy encoding (Eq. 1)."""

from dataclasses import asdict
from itertools import product
import ast
import math
from pathlib import Path
from .io import load_spec
from ..core.hashing import jsonable
from ..legacy_toy.model import Config


def _capacity(values):
    """Build the declared discrete uniform capacity prior (Eqs. 1–19)."""
    return {
        "support": list(values),
        "probabilities": [1 / len(values)] * len(values),
        "public": False,
    }


def toy_hub(config: Config | None = None):
    """
    Encode the frozen prototype network and behavior with source provenance (Eqs. 1–19).
    """
    c = config or Config()
    T = c.horizon
    split = math.ceil(T * 0.4)
    legacy_path = Path(__file__).parents[1] / "legacy_toy/model.py"
    tree = ast.parse(legacy_path.read_text())
    locations = {
        n.name: {
            "source": "airline_market/legacy_toy/model.py",
            "lines": [n.lineno, n.end_lineno],
        }
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name
        in (
            "Config",
            "prices",
            "rival_quotes",
            "market_weights",
            "taste_weights",
            "step",
            "utilities",
            "metrics",
        )
    }
    leg_rows = [
        ("A_OH", "A", "O", "H", (c.own_capacity[0],)),
        ("A_HD", "A", "H", "D", (c.own_capacity[1],)),
        ("B_OH", "B", "O", "H", (c.rival_oh_capacity,)),
        ("B_HD", "B", "H", "D", c.rival_hd_capacities),
    ]
    rows = [
        ("A_OH", "A", "OH", ["A_OH"], 15, 110),
        ("A_OD", "A", "OD", ["A_OH", "A_HD"], 30, 185),
        ("B_OH", "B", "OH", ["B_OH"], 15, 125),
        ("B_HD", "B", "HD", ["B_HD"], 12, 85),
        ("B_OD", "B", "OD", ["B_OH", "B_HD"], 30, 170),
    ]
    products = [
        dict(
            id=j,
            carrier=i,
            market=m,
            legs=legs,
            service_cost=cost,
            reference_fare=ref,
            fare_floor=70,
            fare_cap=300,
            priority=k,
        )
        for k, (j, i, m, legs, cost, ref) in enumerate(rows)
    ]
    rules = {
        "B_OH": {
            "base": 125,
            "scarcity": [{"leg": "B_OH", "threshold": 1, "coefficient": 15}],
            "responses": [
                {"product": "A_OH", "reference": 110, "coefficient": c.rival_response}
            ],
        },
        "B_HD": {
            "base": 85,
            "scarcity": [
                {"leg": "B_HD", "threshold": 1, "coefficient": c.feeder_premium}
            ],
        },
        "B_OD": {
            "base": 170,
            "responses": [
                {"product": "A_OD", "reference": 185, "coefficient": c.rival_response}
            ],
        },
    }
    requests = [
        dict(id="q_" + j, observer="A", target_products=[j], lag_periods=0)
        for j in ("B_OH", "B_OD", "B_HD", "XY")
    ]
    requests.append(
        dict(
            id="q_A_previous",
            observer="B",
            target_products=["A_OH", "A_OD"],
            lag_periods=1,
        )
    )
    direct = ["q_B_OH", "q_B_OD"]
    data = dict(
        schema_version="airline-market-spec/1",
        name="toy_hub",
        description="Exact encoding of the frozen prototype",
        horizon={"n_periods": T},
        airports=[{"code": a} for a in ("O", "H", "D")],
        carriers=[{"code": "A"}, {"code": "B"}],
        legs=[
            dict(id=j, carrier=i, origin=o, dest=d, capacity=_capacity(values))
            for j, i, o, d, values in leg_rows
        ],
        products=products,
        markets=[
            dict(id=m, origin=o, dest=d, in_segment=(m != "HD"))
            for m, o, d in [("OH", "O", "H"), ("HD", "H", "D"), ("OD", "O", "D")]
        ],
        segments=[
            {
                "id": "leisure",
                "classes": [
                    dict(
                        weight=1,
                        alpha=0.025,
                        carrier_pref={"A": 0.0, "B": 0.30},
                        market_value={"OH": 4.0, "HD": 4.0, "OD": 5.0},
                    )
                ],
            },
            {
                "id": "business",
                "classes": [
                    dict(
                        weight=1,
                        alpha=0.012,
                        carrier_pref={"A": 0.25, "B": 0.0},
                        market_value={"OH": 3.5, "HD": 3.5, "OD": 4.5},
                    )
                ],
            },
        ],
        demand=dict(
            mode="single_request_bernoulli",
            arrival_probability={"constant": c.arrival_probability},
            arrival_probability_cap=0.98,
            market_weights={
                "piecewise": [[0, [0.55, 0.40, 0.05]], [split, [0.15, 0.15, 0.70]]]
            },
            segment_weights={"piecewise": [[0, [0.8, 0.2]], [split, [1 - 0.7, 0.7]]]},
            holiday_multiplier=c.holiday_multiplier,
            exogenous_multipliers={"disrupted": 0.70},
        ),
        choice={
            "model": "latent_class_mnl",
            "connection_penalty_rule": "legacy_market_od",
        },
        exogenous=dict(
            states=[
                {"name": "normal", "attributes": {"disrupted": 0}},
                {"name": "disrupted", "attributes": {"disrupted": 1}},
            ],
            initial=[1.0, 0.0],
            transitions={
                str(c.shock_step - 1): [
                    [1 - c.capacity_shock_probability, c.capacity_shock_probability],
                    [0.0, 1.0],
                ]
            },
        ),
        capacity_process={
            "events": [
                {
                    "trigger": {"exo_transition": ["normal", "disrupted"]},
                    "effects": [["A_OH", -1], ["B_HD", -1]],
                }
            ]
        },
        resolution={"penalty_per_cancellation": c.disruption_penalty},
        quotes={"requests": requests},
        pricing=[
            dict(
                carrier="A",
                base={"A_OH": 110, "A_OD": 185},
                time_ramp=15,
                fixed_features=[
                    [
                        {
                            "name": "direct_rival_gap",
                            "params": {"targets": {"A_OH": "B_OH", "A_OD": "B_OD"}},
                        },
                        0.10,
                    ]
                ],
                controllable_features=[
                    {"name": "intercept"},
                    {"name": "connection_indicator"},
                    {"name": "scarcity_sum"},
                ],
                theta_bounds=[[-30, 30], [-30, 30], [0, 15]],
                theta_grid=list(
                    product((-30.0, 0.0, 30.0), (-30.0, 0.0, 30.0), (0.0, 15.0))
                ),
            ),
            dict(carrier="B", base={"B_OH": 125, "B_HD": 85, "B_OD": 170}),
        ],
        controllers=[
            dict(carrier="A", controller="external"),
            dict(
                carrier="B",
                controller="threshold_linear_rule",
                params={"rules": rules, "arm": "B_direct"},
            ),
        ],
        objectives=[
            dict(
                carrier=i,
                profit_scale=c.profit_scale,
                share_weight=c.share_weight,
                share_markets=["OH", "OD"],
            )
            for i in ("A", "B")
        ],
        information=[
            dict(name="direct", observer="A", requests=direct),
            dict(name="linked", observer="A", requests=direct + ["q_B_HD"]),
            dict(name="unrelated", observer="A", requests=direct + ["q_XY"]),
            dict(name="oracle", observer="A", requests=direct, privileged=True),
            dict(name="B_direct", observer="B", requests=["q_A_previous"]),
        ],
        background=[
            dict(
                id="XY",
                carrier="B",
                capacity=_capacity((1, 3)),
                quote_rule={"base": 85, "scarcity_premium": 35, "threshold": 1},
                purchase={"max_prob": 0.7, "intercept": 4.0, "slope": 0.025},
            )
        ],
        timing={
            "mode": "leader_follower",
            "leaders": ["B"],
            "initial_previous_prices": {"A_OH": 110.0, "A_OD": 185.0},
        },
        metadata={
            "legacy_map": locations,
            "legacy_config": asdict(c),
            "connection_penalties": {"OD": 0.5},
            "nested_arms": [["direct", "linked"], ["direct", "unrelated"]],
        },
    )
    return load_spec(data)


def cx_example(kind: str):
    """
    Build the checked correlated/crossed capacity-prior decision diagnostic (Eqs. 1–19).
    """
    data = jsonable(toy_hub())
    data.update(
        name="cx_example_" + kind,
        description="C/X belief compression diagnostic, synthetic",
    )
    data["horizon"] = {"n_periods": 4}
    data["airports"] = [{"code": x} for x in ("O", "H", "D")]
    data["carriers"] = [{"code": "F"}, {"code": "R"}]
    data["legs"] = [
        dict(
            id=i + str(k),
            carrier=i,
            origin=o,
            dest=d,
            capacity=_capacity((1,) if i == "F" else (1, 2)),
        )
        for i in ("F", "R")
        for k, o, d in [(1, "O", "H"), (2, "H", "D")]
    ]
    data["capacity_priors"] = [
        dict(
            legs=["R1", "R2"],
            support=[[1, 1], [2, 2]] if kind == "C" else [[1, 2], [2, 1]],
            probabilities=[0.5, 0.5],
        )
    ]
    markets = [
        ("L1", "O", "H"),
        ("RL1", "O", "H"),
        ("RL2", "H", "D"),
        ("CON", "O", "D"),
    ]
    data["markets"] = [
        dict(id=m, origin=o, dest=d, in_segment=True) for m, o, d in markets
    ]
    prods = [
        ("F_L1", "F", "L1", ["F1"], 150),
        ("F_CON", "F", "CON", ["F1", "F2"], 200),
        ("R_L1", "R", "RL1", ["R1"], 100),
        ("R_L2", "R", "RL2", ["R2"], 100),
        ("R_CON", "R", "CON", ["R1", "R2"], 190),
    ]
    data["products"] = [
        dict(
            id=j,
            carrier=c,
            market=m,
            legs=l,
            service_cost=0,
            fare_floor=1,
            fare_cap=10000,
            reference_fare=p,
            priority=k,
        )
        for k, (j, c, m, l, p) in enumerate(prods)
    ]
    data["segments"] = [
        {
            "id": "captive",
            "classes": [
                dict(
                    weight=1,
                    alpha=0.2,
                    carrier_pref={"R": 50.0, "F": 0.0},
                    market_value={"*": 1000.0},
                )
            ],
        }
    ]
    data["demand"] = dict(
        mode="single_request_bernoulli",
        arrival_probability={"constant": 1.0},
        arrival_probability_cap=1.0,
        market_weights={
            "piecewise": [[t, [float(t == k) for k in range(4)]] for t in range(4)]
        },
        segment_weights={"constant": [1.0]},
    )
    data["choice"] = {"model": "latent_class_mnl"}
    data["exogenous"] = {
        "states": [{"name": "normal"}],
        "initial": [1.0],
        "transitions": {},
    }
    data["capacity_process"] = {}
    data["resolution"] = {}
    data["quotes"] = {"requests": []}
    data["background"] = []
    data["pricing"] = [
        dict(
            carrier="F",
            base={"F_L1": 150, "F_CON": 200},
            controllable_features=[
                {"name": "product_indicator", "params": {"products": ["F_L1"]}}
            ],
            theta_bounds=[[0, 9850]],
            theta_grid=[[0], [9850]],
        ),
        dict(carrier="R", base={p[0]: p[4] for p in prods if p[1] == "R"}),
    ]
    data["controllers"] = [
        {"carrier": "F", "controller": "external"},
        {
            "carrier": "R",
            "controller": "threshold_linear_rule",
            "params": {"rules": {p[0]: {"base": p[4]} for p in prods if p[1] == "R"}},
        },
    ]
    data["objectives"] = [{"carrier": i, "profit_scale": 1.0} for i in ("F", "R")]
    data["information"] = [{"name": "empty", "observer": "F", "requests": []}]
    data["timing"] = {"mode": "leader_follower", "leaders": ["R"]}
    data["metadata"] = {"example": kind}
    return load_spec(data)


def independent_legs():
    """Build the same demand markets with independent rival resources (Eqs. 1–19)."""
    data = jsonable(toy_hub())
    data["name"] = "independent_legs"
    for old, new in [("B_OH", "B_OH_CON"), ("B_HD", "B_HD_CON")]:
        leg = next(l.copy() for l in data["legs"] if l["id"] == old)
        leg["id"] = new
        leg["capacity"] = _capacity((2,))
        data["legs"].append(leg)
    next(p for p in data["products"] if p["id"] == "B_OD")["legs"] = [
        "B_OH_CON",
        "B_HD_CON",
    ]
    # No feeder shares a direct product's resources, so the linked arm is direct.
    next(a for a in data["information"] if a["name"] == "linked")["requests"] = [
        "q_B_OH",
        "q_B_OD",
    ]
    return load_spec(data)
