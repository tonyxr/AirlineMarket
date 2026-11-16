"""A.10/A.5/B0/V9: exact finite-grid pushforward of native class selection.

The uniform53 mark law and its pre-purchase information set are explicit model
assumptions. Native population weights are neither normalized nor substituted.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.exogenous import categorical
from airline_market.sim.passenger_accounts import _binary, _encode
from airline_market.sim.payoff_logit import _ratio
from airline_market.spec.compiled import CompiledMarket

MARK_LAW = "iid-uniform53-before-current-purchase"


@dataclass(frozen=True)
class NativeClassMarkLaw:
    weights: tuple
    details: object


def _grid_pushforward(weights, bits):
    """Count the monotone native kernel's preimages, no grid draw/enumeration.

    Every k/2**bits is exactly binary64 for 1 <= bits <= 53. Bisection delegates
    the native categorical selector; zero-count classes remain explicit.
    """
    weights = tuple(float(w) for w in weights)
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("complete finite nonnegative native weights required")
    if type(bits) is not int or not 1 <= bits <= 53:
        raise ValueError("binary64-exact grid bits must be in 1..53")
    size = 1 << bits
    calls = []

    def selected(k):
        value = categorical(k / size, weights)
        if type(value) is not int or not 0 <= value < len(weights):
            raise RuntimeError("native categorical result outside full class support")
        calls.append((k, value))
        return value

    boundaries = [0]
    for index in range(len(weights) - 1):
        lo, hi = boundaries[-1], size
        while lo < hi:
            mid = (lo + hi) // 2
            if selected(mid) <= index:
                lo = mid + 1
            else:
                hi = mid
        boundaries.append(lo)
    boundaries.append(size)
    rows = []
    for index, (lo, hi) in enumerate(pairwise(boundaries)):
        first = last = None
        if lo < hi:
            first, last = selected(lo), selected(hi - 1)
            if first != index or last != index:
                raise RuntimeError(
                    "native categorical monotone interval witness differs"
                )
        rows.append((index, lo, hi, hi - lo, first, last))
    return tuple(rows), tuple(calls)


def native_class_mark_law(compiled, *, segment_id, mark_law, conditioning, class_limit):
    """Native sampler pushforward under DECLARED pre-purchase uniform53 marks.

    This supplies an explicit class prior only for the declared information set;
    it does not prove seed randomness, mark independence or menu exogeneity.
    """
    if type(compiled) is not CompiledMarket:
        raise TypeError("native CompiledMarket required")
    if (
        mark_law != MARK_LAW
        or not isinstance(conditioning, str)
        or not conditioning.strip()
    ):
        raise ValueError(
            "explicit uniform53 pre-purchase mark law/conditioning required"
        )
    if compiled.spec.choice.model != "latent_class_mnl":
        raise ValueError("finite native class path only; no mixed fallback")
    if segment_id not in compiled.segment_index:
        raise ValueError("existing native segment required")
    segment = compiled.spec.segments[compiled.segment_index[segment_id]]
    if segment.mixed is not None:
        raise ValueError("mixed declarations unsupported on finite class law path")
    if type(class_limit) is not int or class_limit < 1:
        raise ValueError("positive infrastructure class limit required")
    if len(segment.classes) > class_limit:
        raise RuntimeError("ALL classes exceed limit; no partial law or retry")
    raw = tuple(float(c.weight) for c in segment.classes)
    rows, calls = _grid_pushforward(raw, 53)
    size = 1 << 53
    probabilities = tuple((k, Fraction(n, size)) for k, lo, hi, n, first, last in rows)
    exact_raw_mass = sum((Fraction.from_float(w) for w in raw), Fraction())
    return NativeClassMarkLaw(
        probabilities,
        deep_freeze(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "spec_hash": compiled.spec_hash,
                "complete_native_model": jsonable(compiled.spec),
                "segment": segment_id,
                "mark_law": MARK_LAW,
                "conditioning": conditioning,
                "uniform_mark_points": str(size),
                "support": "all integers 0 <= k < 2**53, mark=k/2**53",
                "classes": tuple(
                    {
                        "class_index": k,
                        "native_taste": jsonable(segment.classes[k]),
                        "raw_native_weight": _binary(raw[k]),
                        "integer_interval_start_inclusive": str(lo),
                        "integer_interval_stop_exclusive": str(hi),
                        "grid_points": str(n),
                        "probability": _ratio(Fraction(n, size)),
                        "probability_minus_raw_binary64_weight": _ratio(
                            Fraction(n, size) - Fraction.from_float(raw[k])
                        ),
                        "first_last_native_class_witness": (first, last),
                    }
                    for k, lo, hi, n, first, last in rows
                ),
                "all_native_kernel_evaluations": tuple((str(k), c) for k, c in calls),
                "native_kernel_call_count": len(calls),
                "raw_float_weight_mass": _binary(math.fsum(raw)),
                "exact_raw_binary64_weight_mass": _ratio(exact_raw_mass),
                "exact_raw_binary64_weight_mass_residual": _ratio(exact_raw_mass - 1),
                "exact_grid_probability_mass": _ratio(
                    sum((w for k, w in probabilities), Fraction())
                ),
                "uniform53_and_information_independence_declared_not_authenticated": True,
                "native_generator_seed_joint_law_authenticated": False,
                "menu_independence_or_booking_identity_authenticated": False,
                "native_population_future_law_or_lifecycle_CS_adopted": False,
                "scientific_budget": None,
            }
        ),
    )


def class_mark_law_record(result):
    if type(result) is not NativeClassMarkLaw:
        raise TypeError("NativeClassMarkLaw required")
    return _encode(jsonable(result.details))
