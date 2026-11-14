"""B2/H2.1: exact three-channel fare-spike arithmetic on declared paired inputs.

This component does not generate/authenticate mechanical, forecast or strategic
counterfactuals, select fare weights/quantiles, simulate, infer or test H2.1.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import combinations

from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc

CHANNELS = ("M", "F", "S")
SUBSETS = tuple(frozenset(k) for n in range(4) for k in combinations(CHANNELS, n))


def _number(value):
    if value is None or isinstance(value, Fraction):
        return value
    if type(value) is bool or isinstance(value, float):
        raise TypeError(
            "explicit decimal/rational measurement required, no float precision guess"
        )
    if isinstance(value, Mapping):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError("explicit rational numerator/denominator required")
        if any(
            type(value[k]) is not int and not isinstance(value[k], str)
            for k in ("numerator", "denominator")
        ):
            raise TypeError(
                "rational fields must be integer declarations, no truncation"
            )
        denominator = int(value["denominator"])
        if denominator == 0:
            raise ValueError("nonzero rational denominator required")
        return Fraction(int(value["numerator"]), denominator)
    if not isinstance(value, (str, Decimal, int)):
        raise TypeError("finite decimal, integer, rational or explicit null required")
    try:
        x = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid declared measurement") from error
    if not x.is_finite():
        raise ValueError(
            "nonfinite measurement must remain an explicit unknown upstream"
        )
    return Fraction(x)


def _json_number(value):
    return (
        None
        if value is None
        else {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    )


def _hash(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("declared lowercase SHA256 pin required")
    return value


def _label(subset):
    return "+".join(c for c in CHANNELS if c in subset) or "frozen"


@dataclass(frozen=True)
class FareMetricContract:
    pair_id: str
    cohort_seed: int
    metric_id: str
    input_unit: str
    weighting_id: str
    observation_basis_id: str
    physical_bank_sha256: str
    baseline_run_id: str
    reveal_utc: object
    window_hours: int

    def __post_init__(self):
        for value in (
            self.pair_id,
            self.input_unit,
            self.weighting_id,
            self.observation_basis_id,
            self.baseline_run_id,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "explicit paired metric/weight/support/baseline identifiers required"
                )
        if type(self.cohort_seed) is not int or self.cohort_seed < 0:
            raise ValueError("explicit physical cohort coordinate required")
        if self.metric_id not in {"mean_fare", "p90_fare"}:
            raise ValueError("declare B2 mean-fare or P90-fare metric separately")
        if type(self.window_hours) is not int or self.window_hours != 72:
            raise ValueError("B2 comparison window is exactly 72 elapsed hours")
        _hash(self.physical_bank_sha256)
        object.__setattr__(self, "reveal_utc", utc(self.reveal_utc))

    @property
    def end_utc(self):
        return self.reveal_utc + timedelta(hours=72)

    def contains(self, instant):
        return self.reveal_utc <= utc(instant) < self.end_utc

    @property
    def pair_key(self):
        return (self.pair_id, self.cohort_seed, self.metric_id)

    def record(self):
        return {
            k: v.isoformat() if k == "reveal_utc" else v for k, v in vars(self).items()
        }


@dataclass(frozen=True)
class FareMetricMeasurement:
    contract: FareMetricContract
    run_id: str
    channels: object
    value: object
    source_result_sha256: str
    declared_value_literal: object = field(init=False)

    def __post_init__(self):
        if type(self.contract) is not FareMetricContract:
            raise TypeError("FareMetricContract required")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("explicit source run identifier required")
        _hash(self.source_result_sha256)
        if self.channels is not None:
            if isinstance(self.channels, str):
                raise TypeError(
                    "explicit channel collection required; None is no-shock"
                )
            raw = tuple(self.channels)
            if len(set(raw)) != len(raw) or set(raw) - set(CHANNELS):
                raise ValueError("distinct M/F/S channels required")
            object.__setattr__(self, "channels", frozenset(raw))
        raw = self.value
        number = _number(raw)
        literal = (
            None
            if raw is None
            else dict(raw)
            if isinstance(raw, Mapping)
            else (_json_number(raw) if isinstance(raw, Fraction) else str(raw))
        )
        object.__setattr__(self, "declared_value_literal", deep_freeze(literal))
        object.__setattr__(self, "value", number)

    def record(self):
        return {
            "contract": self.contract.record(),
            "run_id": self.run_id,
            "channels": None
            if self.channels is None
            else [c for c in CHANNELS if c in self.channels],
            "value": _json_number(self.value),
            "declared_value_literal": dict(self.declared_value_literal)
            if isinstance(self.declared_value_literal, Mapping)
            else self.declared_value_literal,
            "source_result_sha256": self.source_result_sha256,
        }

    @classmethod
    def from_record(cls, record):
        out = cls(
            FareMetricContract(**record["contract"]),
            record["run_id"],
            record["channels"],
            record["declared_value_literal"],
            record["source_result_sha256"],
        )
        if out.record()["value"] != record["value"]:
            raise ValueError("declared literal differs from canonical measured value")
        return out


@dataclass(frozen=True)
class SpikeDecomposition:
    baseline: FareMetricMeasurement
    measurements: tuple
    spike_definition: str
    spikes: tuple
    components: tuple
    shares_of_full_spike: tuple
    residual_share: object
    status: str
    missing: tuple

    @property
    def empty_residual(self):
        return dict(self.spikes)[frozenset()]

    @property
    def full_spike(self):
        return dict(self.spikes)[frozenset(CHANNELS)]

    def record(self):
        values = dict(self.spikes)
        complete = self.status == "complete_algebra"
        return {
            "plan_id": "B2/H2.1",
            "contract": self.baseline.contract.record(),
            "spike_definition": self.spike_definition,
            "spike_unit": self.baseline.contract.input_unit
            if self.spike_definition == "level_difference"
            else "ratio",
            "status": self.status,
            "missing_measurements": list(self.missing),
            "baseline": self.baseline.record(),
            "counterfactuals": [m.record() for m in self.measurements],
            "spikes": {_label(k): _json_number(v) for k, v in self.spikes},
            "shapley_components": {k: _json_number(v) for k, v in self.components},
            "frozen_residual": _json_number(self.empty_residual),
            "frozen_declared_metric_difference_is_zero": None
            if self.empty_residual is None
            else self.empty_residual == 0,
            "full_spike": _json_number(self.full_spike),
            "shares_of_full_spike": {
                k: _json_number(v) for k, v in self.shares_of_full_spike
            },
            "frozen_residual_share": _json_number(self.residual_share),
            "ladder": {
                _label(k): _json_number(values[k])
                for k in (
                    frozenset(),
                    frozenset({"M"}),
                    frozenset({"M", "F"}),
                    frozenset(CHANNELS),
                )
            },
            "exact_efficiency_verified": bool(
                complete
                and self.empty_residual + sum(v for _, v in self.components)
                == self.full_spike
            ),
            "source_and_channel_semantics_authenticated": False,
            "declared_hashes_are_not_original_transitive_seals": True,
            "H21_inference_or_full_B2_adoption": False,
            "scientific_budget": None,
        }


def decompose_fare_spike(baseline, measurements, *, spike_definition):
    """Exact B2 formula; residual + sum(phi) = full, including nonzero frozen spike.

    Hash/support/channel declarations are required for pairing, NOT authenticated
    source provenance or economic counterfactual validity. No missing row/value
    is dropped, completed from another context, zero-filled or epsilon-repaired.
    """
    if type(baseline) is not FareMetricMeasurement or baseline.channels is not None:
        raise TypeError("distinct explicit no-shock measurement required")
    if baseline.run_id != baseline.contract.baseline_run_id:
        raise ValueError("no-shock source differs from declared common baseline")
    if spike_definition not in {"level_difference", "relative_change"}:
        raise ValueError(
            "explicit level_difference or relative_change definition required"
        )
    rows = tuple(measurements)
    if any(type(m) is not FareMetricMeasurement for m in rows):
        raise TypeError("explicit measured counterfactual records required")
    if any(m.contract != baseline.contract for m in rows):
        raise ValueError(
            "counterfactual pair/seed/bank/metric/unit/weights/support/window contracts differ"
        )
    if any(m.channels is None for m in rows):
        raise ValueError(
            "no-shock cannot substitute for frozen empty-channel counterfactual"
        )
    if len(rows) != 8 or {m.channels for m in rows} != set(SUBSETS):
        raise ValueError(
            "exactly one measurement for EACH of eight M/F/S subsets required"
        )
    by = {m.channels: m for m in rows}
    ordered = tuple(by[k] for k in SUBSETS)
    missing = tuple(
        (["no_shock"] if baseline.value is None else [])
        + [_label(k) for k in SUBSETS if by[k].value is None]
    )
    bad_ratio = (
        spike_definition == "relative_change"
        and baseline.value is not None
        and baseline.value <= 0
    )
    spikes = []
    for k in SUBSETS:
        v = (
            None
            if baseline.value is None or by[k].value is None or bad_ratio
            else by[k].value - baseline.value
        )
        if v is not None and spike_definition == "relative_change":
            v /= baseline.value
        spikes.append((k, v))
    values = dict(spikes)
    status = (
        "nonpositive_relative_baseline"
        if bad_ratio
        else "incomplete_measurements"
        if missing
        else "complete_algebra"
    )
    components = []
    for channel in CHANNELS:
        v = None
        if status == "complete_algebra":
            v = sum(
                (Fraction(1, 3) if len(k) in (0, 2) else Fraction(1, 6))
                * (values[k | {channel}] - values[k])
                for k in SUBSETS
                if channel not in k
            )
        components.append((channel, v))
    full = values[frozenset(CHANNELS)]
    can_share = status == "complete_algebra" and full != 0
    shares = tuple((c, v / full if can_share else None) for c, v in components)
    residual_share = values[frozenset()] / full if can_share else None
    if status == "complete_algebra":
        assert values[frozenset()] + sum(v for _, v in components) == full
    return SpikeDecomposition(
        baseline,
        ordered,
        spike_definition,
        tuple(spikes),
        tuple(components),
        shares,
        residual_share,
        status,
        missing,
    )


def paired_decompositions(records, *, spike_definition):
    """Keep every declared context/metric, including explicit unknown outcomes.

    No pooled estimates, bootstrap, hypothesis test or scientific n claim. Missing
    structural rows are an input error; explicit null measures remain in output.
    """
    groups = defaultdict(list)
    for record in records:
        if type(record) is not FareMetricMeasurement:
            raise TypeError("FareMetricMeasurement records required")
        groups[record.contract.pair_key].append(record)
    output = []
    for key in sorted(groups):
        rows = groups[key]
        baselines = [m for m in rows if m.channels is None]
        if len(baselines) != 1:
            raise ValueError("each pair/metric needs ONE separate no-shock measurement")
        output.append(
            decompose_fare_spike(
                baselines[0],
                [m for m in rows if m.channels is not None],
                spike_definition=spike_definition,
            )
        )
    return tuple(output)
