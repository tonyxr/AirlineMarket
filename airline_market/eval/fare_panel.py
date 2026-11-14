"""B2/H2.1: saved native regular-posting panel -> declared fare diagnostics.

New producer projection only, no replay/actor/choice/filter/kernel/RNG. Exact
binary64 prices and explicit fixed weights/quantile rule; no primary B2 adoption.
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from airline_market.core.hashing import jsonable, spec_hash
from airline_market.core.snapshot import deep_freeze
from airline_market.eval.spike_decomposition import (
    FareMetricContract,
    FareMetricMeasurement,
    _json_number,
    _number,
)
from airline_market.policy.dated_reserve import UtcBookingClock


@dataclass(frozen=True)
class FarePanelDefinition:
    public_products: tuple
    row_weights: tuple
    weighting_id: str
    price_precision: str
    p90_rule: str

    def __post_init__(self):
        products = tuple(sorted(self.public_products))
        if (
            not products
            or len(set(products)) != len(products)
            or any(not isinstance(p, str) or not p for p in products)
        ):
            raise ValueError("distinct explicit declared-public product IDs required")
        weights = tuple(sorted((t, p, _number(w)) for t, p, w in self.row_weights))
        if len({(t, p) for t, p, w in weights}) != len(weights) or any(
            type(t) is not int or t < 0 or p not in products or w is None or w < 0
            for t, p, w in weights
        ):
            raise ValueError(
                "explicit unique nonnegative period/product weights required"
            )
        if not isinstance(self.weighting_id, str) or not self.weighting_id.strip():
            raise ValueError("explicit weighting identifier required")
        if self.price_precision != "exact_binary64":
            raise ValueError("native precision must explicitly be exact_binary64")
        if self.p90_rule != "weighted_inverse_cdf":
            raise ValueError("explicit supported weighted_inverse_cdf rule required")
        object.__setattr__(self, "public_products", products)
        object.__setattr__(self, "row_weights", weights)

    def record(self, compiled, clock):
        T = compiled.spec.horizon.n_periods
        if not isinstance(clock, UtcBookingClock) or len(clock.boundaries) != T + 1:
            raise ValueError("explicit native-period clock plus terminal required")
        if set(self.public_products) - set(compiled.product_index):
            raise ValueError("declared-public product outside compiled source schema")
        expected = {(t, p) for t in range(T) for p in self.public_products}
        if {(t, p) for t, p, w in self.row_weights} != expected:
            raise ValueError(
                "weights must cover EVERY period/selected-product coordinate"
            )
        return {
            "format": "B2-native-regular-posting-panel/1",
            "source_product_order": [p.id for p in compiled.spec.products],
            "selected_products": [
                {
                    "product": p,
                    "market": compiled.spec.products[compiled.product_index[p]].market,
                }
                for p in self.public_products
            ],
            "clock_utc": [t.isoformat() for t in clock.boundaries],
            "weights": [
                {"period": t, "product": p, "weight": _json_number(w)}
                for t, p, w in self.row_weights
            ],
            "weighting_id": self.weighting_id,
            "price_precision": self.price_precision,
            "p90_rule": self.p90_rule,
            "meaning": "declared public-mask regular native posting opportunities; not sales/bookability/exposure",
        }

    def basis_id(self, compiled, clock):
        return "B2-native-regular-posting-panel/1:" + spec_hash(
            self.record(compiled, clock)
        )


def panel_contract(compiled, clock, definition, **pair_fields):
    """Bind actual grid/weights/precision/convention digest; caller declares units/pair."""
    if (
        "observation_basis_id" in pair_fields
        or "weighting_id" in pair_fields
        or "metric_id" in pair_fields
    ):
        raise ValueError("panel derives support/weight/mean anchor identifiers")
    return FareMetricContract(
        metric_id="mean_fare",
        weighting_id=definition.weighting_id,
        observation_basis_id=definition.basis_id(compiled, clock),
        **pair_fields,
    )


def _price_record(value):
    if value is None:
        return {
            "status": "missing",
            "binary64_hex": None,
            "binary64_bits": None,
            "repr": None,
            "value": None,
        }
    if type(value) is bool or not isinstance(value, (int, float)):
        raise TypeError("native binary64 price vector required")
    x = float(value)
    return {
        "status": "finite" if math.isfinite(x) else "nonfinite",
        "binary64_hex": x.hex(),
        "binary64_bits": struct.pack(">d", x).hex(),
        "repr": repr(x),
        "value": _json_number(Fraction.from_float(x)) if math.isfinite(x) else None,
    }


def _read_price(record):
    if record["status"] == "missing":
        if any(
            record[k] is not None
            for k in ("binary64_hex", "binary64_bits", "repr", "value")
        ):
            raise ValueError("missing price has inconsistent retained fields")
        return None
    x = struct.unpack(">d", bytes.fromhex(record["binary64_bits"]))[0]
    expected = _price_record(x)
    if expected != record:
        raise ValueError("retained native price/precision fields disagree")
    return _number(record["value"])


def capture_native_posted_panel(
    engine, clock, definition, contract, *, run_id, channels, destination
):
    """Project NEW native trace prices once; never call post_actions/step/quote/RNG.

    Selection is caller-declared public visibility, not authenticated public truth.
    No purchases, events, availability, private menus or hidden stock are exported.
    Terminal is native state only; actual producer completion is checked at read.
    """
    c = engine.compiled
    definition_record = definition.record(c, clock)
    if (
        contract.metric_id != "mean_fare"
        or contract.observation_basis_id != definition.basis_id(c, clock)
        or contract.weighting_id != definition.weighting_id
    ):
        raise ValueError("source contract differs from actual panel definition")
    if engine.tape.seed != contract.cohort_seed:
        raise ValueError("native tape seed differs from declared paired coordinate")
    # Validate source labels/precision through the existing frozen measurement schema.
    identity = FareMetricMeasurement(contract, run_id, channels, None, "0" * 64)
    target = Path(destination)
    if target.exists():
        raise FileExistsError("retain existing panel; new output required")
    rows = []
    previous = -1
    for row in engine.trace():
        t = row["period"]
        if type(t) is not int or not previous < t < c.spec.horizon.n_periods:
            raise ValueError("native trace periods must be strictly ordered and unique")
        previous = t
        prices = row["prices"]
        if len(prices) != len(c.spec.products):
            raise ValueError("native price vector/source schema length differs")
        rows.append(
            {
                "period": t,
                "prices": {
                    p: _price_record(prices[c.product_index[p]])
                    for p in definition.public_products
                },
            }
        )
    payload = {
        "plan_id": "B2/H2.1",
        "format": "B2-native-regular-posting-panel/1",
        "run_id": run_id,
        "channels": identity.record()["channels"],
        "contract": contract.record(),
        "definition": definition_record,
        "compiled_spec_sha256": c.spec_hash,
        "native_terminal": bool(engine.terminal()),
        "rows": rows,
        "analyst_only": True,
        "public_visibility_authenticated": False,
        "physical_bank_pin_is_caller_declared": True,
    }
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    target.write_bytes(raw)
    return {
        "posted_panel_path": str(target),
        "posted_panel_sha256": hashlib.sha256(raw).hexdigest(),
    }


@dataclass(frozen=True)
class SavedFarePanelSource:
    panel_path: object
    results_path: object
    receipt_path: object


@dataclass(frozen=True)
class FarePanelMetrics:
    measurements: tuple
    rows: tuple
    reasons: tuple
    provenance: object
    total_weight: object
    unknown_positive_weight: object
    window_covered: bool

    def record(self):
        return {
            "plan_id": "B2/H2.1",
            "status": "measured_declared_panel"
            if not self.reasons
            else "limited_declared_panel",
            "measurements": [m.record() for m in self.measurements],
            "rows": [jsonable(r) for r in self.rows],
            "reasons": list(self.reasons),
            "provenance": jsonable(self.provenance),
            "total_declared_window_weight": _json_number(self.total_weight),
            "unknown_positive_weight": _json_number(self.unknown_positive_weight),
            "native_clock_covers_72h": self.window_covered,
            "economic_channels_or_primary_B2_metric_adopted": False,
            "scientific_budget": None,
        }


def read_saved_fare_panel(
    source, compiled, clock, definition, contract, *, run_id, channels
):
    """Verify current receipt->results->panel byte links and derive declared diagnostics.

    Completion/link verification is NOT original-time provenance, channel/CRN or
    public-visibility authentication. Noncompleted producers yield null metrics,
    even if a prefix covers the window; no crashed-run salvage is performed.
    """
    if type(source) is not SavedFarePanelSource:
        raise TypeError("SavedFarePanelSource required")
    raw = Path(source.panel_path).read_bytes()
    panel_sha = hashlib.sha256(raw).hexdigest()
    result_raw = Path(source.results_path).read_bytes()
    result_sha = hashlib.sha256(result_raw).hexdigest()
    receipt_raw = Path(source.receipt_path).read_bytes()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    result = json.loads(result_raw)
    receipt = json.loads(receipt_raw)
    panel = json.loads(raw)
    if (
        receipt["results_sha256"] != result_sha
        or receipt["status"] != result["status"]
        or receipt["config_hash"] != result["config_hash"]
    ):
        raise ValueError("producer receipt/results byte or identity link differs")
    completed = result["status"] == "completed"
    bound = result.get("posted_panel_sha256") == panel_sha and result.get(
        "posted_panel_path"
    ) == str(source.panel_path)
    if completed and not bound:
        raise ValueError("completed producer does not bind the actual panel bytes/path")
    identity = FareMetricMeasurement(contract, run_id, channels, None, result_sha)
    expected_definition = definition.record(compiled, clock)
    if (
        contract.metric_id != "mean_fare"
        or contract.observation_basis_id != definition.basis_id(compiled, clock)
        or contract.weighting_id != definition.weighting_id
    ):
        raise ValueError("pair contract differs from exact declared panel basis")
    if (
        panel["contract"] != contract.record()
        or panel["definition"] != expected_definition
        or panel["compiled_spec_sha256"] != compiled.spec_hash
        or panel["run_id"] != run_id
        or panel["channels"] != identity.record()["channels"]
        or panel["format"] != "B2-native-regular-posting-panel/1"
    ):
        raise ValueError("panel/source/clock/compiled/pair/channel declarations differ")
    if type(panel["native_terminal"]) is not bool:
        raise ValueError("explicit boolean native terminal marker required")
    T = compiled.spec.horizon.n_periods
    native = {}
    for row in panel["rows"]:
        t = row["period"]
        if (
            type(t) is not int
            or not 0 <= t < T
            or t in native
            or (native and t <= max(native))
        ):
            raise ValueError("saved native periods are duplicated/outside/unsorted")
        if set(row["prices"]) != set(definition.public_products):
            raise ValueError("saved panel public mask differs")
        native[t] = {p: (r, _read_price(r)) for p, r in row["prices"].items()}
    periods = tuple(t for t in range(T) if contract.contains(clock.at(t)))
    weights = {(t, p): w for t, p, w in definition.row_weights}
    rows = []
    total = unknown = Fraction()
    weighted = []
    for t in periods:
        for p in definition.public_products:
            price = native.get(t, {}).get(p)
            value = None if price is None else price[1]
            weight = weights[t, p]
            total += weight
            if value is None:
                unknown += weight
            elif weight > 0:
                weighted.append((value, weight))
            rows.append(
                deep_freeze(
                    {
                        "period": t,
                        "instant_utc": clock.at(t).isoformat(),
                        "product": p,
                        "weight": _json_number(weight),
                        "value": _json_number(value),
                        "source_status": "missing_native_period"
                        if price is None
                        else price[0]["status"],
                        "binary64_hex": None
                        if price is None
                        else price[0]["binary64_hex"],
                        "binary64_bits": None
                        if price is None
                        else price[0]["binary64_bits"],
                        "native_repr": None if price is None else price[0]["repr"],
                    }
                )
            )
    covered = clock.at(0) <= contract.reveal_utc and clock.at(T) >= contract.end_utc
    reasons = []
    if not completed:
        reasons.append("producer_not_completed")
    if not bound:
        reasons.append("panel_not_bound_by_results")
    if not panel["native_terminal"]:
        reasons.append("native_trace_not_terminal")
    if set(native) != set(range(T)):
        reasons.append("missing_native_periods")
    if not covered:
        reasons.append("clock_does_not_cover_full_72h")
    if not total:
        reasons.append("zero_declared_window_weight")
    if unknown:
        reasons.append("unknown_positive_weight_price")
    mean = p90 = None
    if not reasons:
        mean = sum(value * weight for value, weight in weighted) / total
        cumulative = Fraction()
        for value, weight in sorted(weighted):
            cumulative += weight
            if cumulative >= Fraction(9, 10) * total:
                p90 = value
                break
        assert p90 is not None
    measurements = tuple(
        FareMetricMeasurement(
            replace(contract, metric_id=metric), run_id, channels, value, result_sha
        )
        for metric, value in (("mean_fare", mean), ("p90_fare", p90))
    )
    provenance = deep_freeze(
        {
            "panel_path": str(source.panel_path),
            "results_path": str(source.results_path),
            "receipt_path": str(source.receipt_path),
            "panel_sha256": panel_sha,
            "results_sha256": result_sha,
            "receipt_sha256": receipt_sha,
            "producer_status": result["status"],
            "receipt_result_link_verified": True,
            "results_panel_link_verified": bound,
            "original_time_provenance_authenticated": False,
            "physical_CRN_and_economic_channels_authenticated": False,
            "public_visibility_authenticated": False,
            "metrics_are_new_derived_artifacts": True,
        }
    )
    return FarePanelMetrics(
        measurements, tuple(rows), tuple(reasons), provenance, total, unknown, covered
    )
