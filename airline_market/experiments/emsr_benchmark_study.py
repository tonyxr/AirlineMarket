"""Published four-class EMSR-b inputs, independent quantiles and native thresholds."""

import math
import sys
from pathlib import Path
from statistics import NormalDist

from airline_market.agents.base import ControllerContext
from airline_market.agents.emsr import EMSRbController, emsr_b_protection
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.pomdp.observation import Observation
from airline_market.sim.state import Booking
from airline_market.spec import compile_market, dump_spec, load_spec


def independent_reference(fares, means, stds, capacity):
    """Full-precision normal formula; no SciPy/native-helper dependency."""
    if (
        len(fares) != 4
        or len(means) != 4
        or len(stds) != 4
        or any(not math.isfinite(float(x)) for row in [fares, means, stds] for x in row)
        or any(fares[k] <= fares[k + 1] for k in range(3))
        or any(x <= 0 for x in fares + means + stds)
        or capacity <= 0
    ):
        raise ValueError(
            "four strictly descending positive classes/means/stds required"
        )
    rows, levels = [], [0.0]
    for k in range(1, 4):
        mean = math.fsum(means[:k])
        variance = math.fsum(x * x for x in stds[:k])
        average = math.fsum(f * m for f, m in zip(fares[:k], means[:k])) / mean
        probability = 1 - fares[k] / average
        if not 0 < probability < 1:
            raise ValueError("reference percentile must be interior; no epsilon repair")
        quantile = NormalDist().inv_cdf(probability)
        raw = mean + math.sqrt(variance) * quantile
        level = min(float(capacity), max(0.0, raw))
        rows.append(
            {
                "higher_classes": k,
                "mean": mean,
                "variance": variance,
                "weighted_fare": average,
                "quantile_probability": probability,
                "normal_quantile": quantile,
                "raw_protection_seats": raw,
                "capacity_clipped_protection_seats": level,
            }
        )
        levels.append(level)
    return levels, rows


def permitted_observation(capacity, stock):
    """Coherent public own ledger, without an engine/world/tape or hidden state."""
    if (
        type(capacity) is not int
        or type(stock) is not int
        or not 0 <= stock <= capacity
    ):
        raise ValueError("invalid integer inventory observation")
    bookings = tuple(
        Booking(k, 0, 0, 527.0, k, -1, -1, -1) for k in range(capacity - stock)
    )
    return Observation(
        t=capacity,
        carrier=0,
        arm="direct",
        own_capacity=(capacity,),
        own_bookings=bookings,
        own_cash=(capacity - stock) * 527.0,
        own_refunds=0.0,
        own_previous_prices=(527.0,),
        own_posted_prices=(527.0,),
        public_exo={"state:normal": 1.0, "capacity/A_OD": float(capacity)},
        quotes=(),
        own_inventory=(stock,),
        privileged=None,
    )


def controller_fingerprint(controller):
    return schema_hash(
        {
            "attributes": {
                k: v for k, v in vars(controller).items() if k not in {"ctx", "rng"}
            },
            "memory": controller.snapshot(),
            "rng": controller.rng.bit_generator.state,
            "compiled_spec_hash": controller.ctx.compiled.spec_hash,
        }
    )


def capture_actual_thresholds(controller, obs):
    """Read a native return-frame local; never modify code, globals or frame locals."""
    previous = sys.getprofile()
    captures = []
    before = controller_fingerprint(controller)
    observation_hash = schema_hash(obs)

    def profile(frame, event, argument):
        if previous is not None:
            previous(frame, event, argument)
        if (
            frame.f_code is EMSRbController.act.__code__
            and event == "return"
            and argument is not None
        ):
            captures.append(tuple(float(x) for x in frame.f_locals["protection"]))

    try:
        sys.setprofile(profile)
        action = controller.act(obs)
    finally:
        sys.setprofile(previous)
    if len(captures) != 1 or len(captures[0]) != 4:
        raise AssertionError("native single-leg/product threshold capture changed")
    if before != controller_fingerprint(controller) or observation_hash != schema_hash(
        obs
    ):
        raise AssertionError("threshold inspection mutated controller/RNG/observation")
    plain = controller.act(obs)
    if plain != action or before != controller_fingerprint(controller):
        raise AssertionError("profiled/unprofiled native action or state differs")
    return action, captures[0]


def build_controller(c, p):
    if (
        len(c.spec.carriers) != 1
        or len(c.spec.legs) != 1
        or len(c.spec.products) != 1
        or c.spec.products[0].legs != (c.spec.legs[0].id,)
        or c.spec.horizon.n_periods < p["capacity"] + 1
        or c.spec.legs[0].capacity.support != (p["capacity"],)
        or not c.spec.legs[0].capacity.public
    ):
        raise ValueError("single-public-leg/product context contract changed")
    prod = c.spec.products[0]
    if prod.fare_floor > min(p["fares"]) or prod.fare_cap < max(p["fares"]):
        raise ValueError("native fare clipping would change the published ladder")
    controller = EMSRbController(
        ladders={prod.id: p["fares"]},
        forecast={prod.id: {"means": p["means"], "stds": p["stds"]}},
    )
    controller.reset(
        ControllerContext(
            c, 0, c.spec.pricing[0], c.spec.objectives[0], c.spec.information[0]
        ),
        StreamRegistry(773, "planning/V4/EMSR-b-public-audit").generator(),
    )
    return controller


def run(cell, context):
    p = cell["parameters"]
    for name, key in [
        ("source_spec", "source_spec_sha256"),
        ("native_source", "native_source_sha256"),
        ("design_registration", "design_registration_sha256"),
    ]:
        if file_hash(Path(p[name])) != p[key]:
            raise ValueError("frozen EMSR input changed: " + name)
    c = compile_market(load_spec(Path(p["source_spec"])))
    if c.spec_hash != p["compiled_spec_hash"]:
        raise ValueError("EMSR context compiled hash changed")
    dump_spec(c.spec, context.directory / "spec.json")
    reference, derivation = independent_reference(
        p["fares"], p["means"], p["stds"], p["capacity"]
    )
    helper = list(
        map(float, emsr_b_protection(p["fares"], p["means"], p["stds"], p["capacity"]))
    )
    controller = build_controller(c, p)
    current = {
        "stage": "public_inventory_sweep",
        "observations_completed": 0,
        "observations_total": p["capacity"] + 1,
        "simulation_cohorts": 0,
    }
    context.counters = lambda: dict(current)
    rows = []
    for stock in range(p["capacity"] + 1):
        obs = permitted_observation(p["capacity"], stock)
        action, actual = capture_actual_thresholds(controller, obs)
        eligible = [k for k, level in enumerate(reference) if stock > level]
        expected = p["fares"][max(eligible) if eligible else 0]
        error = max(abs(a - b) for a, b in zip(reference, actual))
        rows.append(
            {
                "stock": stock,
                "permitted_observation": jsonable(obs),
                "reference_protection_seats": reference,
                "native_actual_protection_seats": actual,
                "threshold_max_abs_error": error,
                "native_fare": action.prices[0],
                "independent_fare": expected,
                "action_match": action.prices[0] == expected,
                "observation_controller_RNG_unchanged": True,
                "profile_restored": True,
            }
        )
        current["observations_completed"] = stock + 1
    helper_error = max(abs(a - b) for a, b in zip(reference, helper))
    actual_error = max(x["threshold_max_abs_error"] for x in rows)
    published_raw = [round(x["raw_protection_seats"], 1) for x in derivation]
    passed = (
        helper_error <= p["tolerance"]
        and actual_error <= p["tolerance"]
        and all(x["action_match"] for x in rows)
        and published_raw == p["published_one_decimal_EMSR"]
    )
    atomic_json(
        context.directory / "independent_reference.json",
        {
            "protection_seats": reference,
            "derivation": derivation,
            "published_raw_rounded_to_one_decimal": published_raw,
        },
    )
    atomic_json(context.directory / "controller_rows.json", rows)
    atomic_json(
        context.directory / "helper_comparison.json",
        {
            "helper_protection_seats": helper,
            "independent_protection_seats": reference,
            "max_abs_error": helper_error,
        },
    )
    result = {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "capacity": p["capacity"],
        "controller_observations": len(rows),
        "simulation_cohorts": 0,
        "reference_protection_seats": reference,
        "helper_max_abs_error": helper_error,
        "actual_controller_max_abs_error": actual_error,
        "action_mismatches": sum(not x["action_match"] for x in rows),
        "EMSR_fourclass_case_pass": passed,
        "full_V4_pass": None,
        "scientific_budget": None,
        "test_seeds_opened": False,
        "scope": "Published four-class Gaussian forecast-input algorithm verification, full-precision independent normal calculation versus helper and actual controller thresholds. Not native Gaussian-demand simulation, Littlewood study, learning or full V4.",
    }
    names = [
        "spec.json",
        "independent_reference.json",
        "controller_rows.json",
        "helper_comparison.json",
    ]
    result["artifact_hashes"] = {
        name: file_hash(context.directory / name) for name in names
    }
    atomic_json(context.directory / "emsr_fourclass.json", result)
    return {
        **result,
        "summary_sha256": file_hash(context.directory / "emsr_fourclass.json"),
    }
