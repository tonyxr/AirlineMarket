"""Poisson hub instances with 2–4 carriers for F2/F3/F6 (plan Appendix B).

This extends the synthetic catalog, not the empirical calibration. Each
carrier owns its legs; connecting products consume both legs (Eq. 12).
Demand intensity is proportional to prior-expected seats, independent of
the realized private capacity draw (Eq. 5).
"""

from copy import deepcopy
from dataclasses import replace

from ..core.hashing import jsonable
from ..spec import compile_market, load_spec
from ..spec.information import derive_information_arms
from ..spec.model import QuoteSpec, ShoppingRequestSpec
from .generators import hub_feeder_family


def scaling_network(params: dict | None = None):
    """Build the declared Poisson oligopoly without changing the exact family."""
    p = dict(params or {})
    allowed = {
        "n_spokes",
        "n_carriers",
        "periods",
        "capacity",
        "capacity_support",
        "connecting_share",
        "requests_per_initial_seat",
        "lag_periods",
        "controller_mode",
        "share_weight",
        "name",
    }
    if set(p) - allowed:
        raise ValueError(f"unknown scaling parameters: {sorted(set(p) - allowed)}")
    spokes = int(p.get("n_spokes", 8))
    carriers = int(p.get("n_carriers", 2))
    periods = int(p.get("periods", 20))
    requests_per_seat = float(p.get("requests_per_initial_seat", 1.2))
    mode = p.get("controller_mode", "threshold_rivals")
    if carriers not in (2, 3, 4) or spokes < 1 or periods < 2:
        raise ValueError(
            "scaling catalog requires 2–4 carriers, positive spokes and T>=2"
        )
    if requests_per_seat < 0 or mode not in ("threshold_rivals", "external_all"):
        raise ValueError("invalid arrival scale or controller mode")
    base = hub_feeder_family(
        {
            "n_spokes": spokes,
            "periods": periods,
            "capacity": int(p.get("capacity", 3)),
            "capacity_support": p.get("capacity_support", [1, 3]),
            "connecting_share": float(p.get("connecting_share", 0.5)),
            "share_weight": float(p.get("share_weight", 0)),
        }
    )
    data = jsonable(base)
    codes = list("ABCD"[:carriers])
    templates = {
        "legs": [deepcopy(leg) for leg in data["legs"] if leg["carrier"] == "B"],
        "products": [
            deepcopy(product)
            for product in data["products"]
            if product["carrier"] == "B"
        ],
        "pricing": deepcopy(data["pricing"][1]),
        "controller": deepcopy(data["controllers"][1]),
        "objective": deepcopy(data["objectives"][1]),
    }

    def rename(value, code):
        """Rename only B-owned identifiers in the copied rival template."""
        if isinstance(value, dict):
            return {
                rename(key, code): rename(item, code) for key, item in value.items()
            }
        if isinstance(value, list):
            return [rename(item, code) for item in value]
        if isinstance(value, str):
            return (
                code
                if value == "B"
                else code + value[1:]
                if value.startswith("B_")
                else value
            )
        return value

    data["carriers"] = [
        {"code": code, "name": f"Synthetic carrier {code}"} for code in codes
    ]
    for code in codes[2:]:
        data["legs"].extend(rename(templates["legs"], code))
        products = rename(templates["products"], code)
        for product in products:
            product["priority"] = len(data["products"])
            data["products"].append(product)
        data["pricing"].append(rename(templates["pricing"], code))
        data["controllers"].append(rename(templates["controller"], code))
        data["objectives"].append(rename(templates["objective"], code))
    # Every carrier can be overridden by the same three-dimensional coefficient
    # policy during F3. The focal's existing direct-gap term is retained.
    for pricing in data["pricing"][1:]:
        for field in ("controllable_features", "theta_bounds", "theta_grid"):
            pricing[field] = deepcopy(data["pricing"][0][field])
    for segment in data["segments"]:
        for taste in segment["classes"]:
            for code in codes[2:]:
                taste["carrier_pref"][code] = taste["carrier_pref"].get("B", 0.0)
    if mode == "external_all":
        data["controllers"] = [
            {"carrier": code, "controller": "external"} for code in codes
        ]
    expected_seats = sum(
        sum(
            k * probability
            for k, probability in zip(
                leg["capacity"]["support"], leg["capacity"]["probabilities"]
            )
        )
        for leg in data["legs"]
    )
    expected_requests = requests_per_seat * expected_seats
    # Use the same explicit OH/HD/connecting mixture as the Bernoulli family.
    connecting = float(p.get("connecting_share", 0.5))
    weights = [
        connecting / spokes**2
        if market["id"].startswith("OD")
        else (1 - connecting) / (2 * spokes)
        for market in data["markets"]
    ]
    data["demand"] = {
        "mode": "poisson",
        "intensity": {
            market["id"]: {"constant": expected_requests * weight / periods}
            for market, weight in zip(data["markets"], weights)
        },
        "segment_weights": data["demand"]["segment_weights"],
    }
    background = deepcopy(data["background"][0])
    data["background"] = [
        {**deepcopy(background), "id": f"XY_{code}_{k}", "carrier": code}
        for code in codes[1:]
        for k in range(spokes)
    ]
    data["name"] = p.get("name", f"poisson_hub_s{spokes}_c{carriers}_T{periods}")
    data["description"] = (
        "Synthetic asymmetric hub; Poisson demand and private rival feeder capacities; uncalibrated."
    )
    data["timing"] = {"mode": "simultaneous", "leaders": []}
    data["quotes"] = {"requests": []}
    data["information"] = []
    data["metadata"] = {
        "generator": {"name": "scaling_network", "params": p},
        "assumptions": {
            "calibrated": False,
            "preferences_and_price_sensitivity": "Inherited synthetic two-segment tastes; additional rivals inherit B's preferences.",
            "arrival_scale": "Potential requests per prior-expected initial seat; independent of realized private capacity.",
            "requests_per_initial_seat": requests_per_seat,
            "expected_initial_seats": expected_seats,
            "expected_requests_per_cohort": expected_requests,
            "capacity": "Fixed during each cohort; private stochastic initial rival feeder seats.",
        },
    }
    spec = load_spec(data)
    compiled = compile_market(spec)
    requests, information, shortfalls = [], [], {}
    for code in codes:
        arms = derive_information_arms(compiled, code)
        scoped = {}
        for arm in arms.values():
            for old_id in arm.requests:
                target = old_id.removeprefix("q_")
                scoped[old_id] = f"q_{code}_{target}"
        requests.extend(
            ShoppingRequestSpec(
                id=new_id,
                observer=code,
                target_products=(old_id.removeprefix("q_"),),
                lag_periods=int(p.get("lag_periods", 1)),
            )
            for old_id, new_id in scoped.items()
        )
        information.extend(
            replace(arm, requests=tuple(scoped[qid] for qid in arm.requests))
            for arm in arms.values()
        )
        shortfalls[code] = max(
            0, len(arms["linked"].requests) - len(arms["unrelated"].requests)
        )
    return replace(
        spec,
        quotes=QuoteSpec(requests=tuple(requests)),
        information=tuple(information),
        metadata={**spec.metadata, "placebo_shortfall_by_observer": shortfalls},
    )
