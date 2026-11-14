"""B2/H2.1: exact paired Shapley arithmetic, residuals, missingness and provenance scope."""

import hashlib
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

if os.environ.get("B2_SPIKE_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_spike_decomposition", os.environ["B2_SPIKE_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.eval import spike_decomposition as module


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def contract(**changes):
    data = {
        "pair_id": "unit-context",
        "cohort_seed": 31,
        "metric_id": "mean_fare",
        "input_unit": "declared model fare units",
        "weighting_id": "unit fixed explicit weights",
        "observation_basis_id": "unit identical declared support",
        "physical_bank_sha256": sha("unit common potential tape manifest"),
        "baseline_run_id": "unit-noshock",
        "reveal_utc": "2026-03-08T00:00:00Z",
        "window_hours": 72,
    }
    data.update(changes)
    return module.FareMetricContract(**data)


def measurement(c, channels, value, *, run=None):
    name = (
        "no-shock"
        if channels is None
        else "+".join(x for x in module.CHANNELS if x in channels) or "frozen"
    )
    return module.FareMetricMeasurement(
        c,
        run or (c.baseline_run_id if channels is None else "unit-" + name),
        channels,
        value,
        sha("unit source " + name),
    )


def inputs(*, c=None, base=100, residual=7, coefficients=None):
    c = c or contract()
    coefficients = coefficients or {
        frozenset("M"): 10,
        frozenset("F"): -2,
        frozenset("S"): 5,
        frozenset("MF"): 4,
        frozenset("MS"): -6,
        frozenset("FS"): 8,
        frozenset("MFS"): 3,
    }
    rows = []
    for k in module.SUBSETS:
        value = (
            Fraction(base)
            + Fraction(residual)
            + sum(Fraction(v) for term, v in coefficients.items() if term <= k)
        )
        rows.append(measurement(c, k, value))
    return measurement(c, None, base), tuple(rows)


def result(**kwargs):
    a, b = inputs(**kwargs)
    return module.decompose_fare_spike(a, b, spike_definition="level_difference")


def test_interactions_and_nonzero_frozen_residual_match_independent_closed_form():
    r = result()
    assert dict(r.components) == {"M": 10, "F": 5, "S": 7}
    assert r.empty_residual == 7 and r.full_spike == 29
    assert sum(v for c, v in r.components) == 22
    assert r.empty_residual + sum(v for c, v in r.components) == r.full_spike
    assert dict(r.shares_of_full_spike) == {
        "M": Fraction(10, 29),
        "F": Fraction(5, 29),
        "S": Fraction(7, 29),
    }
    assert r.residual_share == Fraction(7, 29)
    assert r.record()["exact_efficiency_verified"] is True


def test_reference_average_of_all_six_ordered_marginal_paths():
    a, rows = inputs()
    values = {r.channels: r.value - a.value for r in rows}
    expect = {c: Fraction() for c in module.CHANNELS}
    for order in permutations(module.CHANNELS):
        active = frozenset()
        for c in order:
            expect[c] += (values[active | {c}] - values[active]) / 6
            active = active | {c}
    assert dict(result().components) == expect


def test_sequential_ladder_keeps_frozen_M_MF_MFS_without_replacing_shapley():
    r = result()
    record = r.record()
    spikes = {
        k: Fraction(int(v["numerator"]), int(v["denominator"]))
        for k, v in record["ladder"].items()
    }
    assert spikes == {"frozen": 7, "M": 17, "M+F": 19, "M+F+S": 29}
    assert [
        spikes["M"] - spikes["frozen"],
        spikes["M+F"] - spikes["M"],
        spikes["M+F+S"] - spikes["M+F"],
    ] == [10, 2, 10]
    assert dict(r.components) == {"M": 10, "F": 5, "S": 7}


def test_signed_zero_and_negative_full_spikes_are_retained_without_clipping():
    r = result(
        residual=0,
        coefficients={frozenset("M"): -3, frozenset("F"): 0, frozenset("S"): 1},
    )
    assert dict(r.components) == {"M": -3, "F": 0, "S": 1} and r.full_spike == -2
    assert dict(r.shares_of_full_spike) == {
        "M": Fraction(3, 2),
        "F": 0,
        "S": Fraction(-1, 2),
    }
    assert r.record()["status"] == "complete_algebra"


def test_zero_full_spike_keeps_canceling_nonzero_components_and_null_shares():
    r = result(
        residual=0,
        coefficients={frozenset("M"): 3, frozenset("F"): -4, frozenset("S"): 1},
    )
    assert r.full_spike == 0 and dict(r.components) == {"M": 3, "F": -4, "S": 1}
    assert (
        all(v is None for _, v in r.shares_of_full_spike) and r.residual_share is None
    )
    assert r.record()["exact_efficiency_verified"]


def test_constant_shift_changes_residual_not_channel_components():
    a = result(residual=7)
    b = result(residual=Fraction(123456789, 1000000))
    assert a.components == b.components
    assert b.full_spike - a.full_spike == b.empty_residual - a.empty_residual


def test_dummy_symmetry_and_exact_fine_precision_are_context_independent():
    with localcontext() as context:
        context.prec = 6
        r = result(
            base=Decimal("100.12345678901234567890123456789"),
            residual=0,
            coefficients={
                frozenset("M"): Fraction(1, 10**40),
                frozenset("F"): Fraction(1, 10**40),
                frozenset("MF"): Fraction(2, 10**40),
            },
        )
    assert dict(r.components) == {
        "M": Fraction(2, 10**40),
        "F": Fraction(2, 10**40),
        "S": 0,
    }
    assert r.baseline.declared_value_literal == "100.12345678901234567890123456789"


def test_relative_change_definition_is_explicit_and_preserves_residual_identity():
    a, rows = inputs()
    r = module.decompose_fare_spike(a, rows, spike_definition="relative_change")
    assert dict(r.components) == {
        "M": Fraction(1, 10),
        "F": Fraction(1, 20),
        "S": Fraction(7, 100),
    }
    assert r.empty_residual == Fraction(7, 100) and r.full_spike == Fraction(29, 100)
    assert (
        r.record()["spike_unit"] == "ratio" and r.record()["exact_efficiency_verified"]
    )
    assert dict(r.shares_of_full_spike) == dict(result().shares_of_full_spike)


@pytest.mark.parametrize("baseline", [0, -10])
def test_nonpositive_relative_denominator_stays_null_without_epsilon_or_zero_fill(
    baseline,
):
    a, rows = inputs(base=baseline)
    r = module.decompose_fare_spike(a, rows, spike_definition="relative_change")
    assert r.status == "nonpositive_relative_baseline"
    assert all(v is None for _, v in r.spikes) and all(
        v is None for _, v in r.components
    )
    assert (
        r.record()["baseline"]["value"] is not None
        and not r.record()["exact_efficiency_verified"]
    )


@pytest.mark.parametrize("which", [None, frozenset(), frozenset("F"), frozenset("MFS")])
def test_explicit_unknown_keeps_every_record_and_does_not_salvage_partial_shapley(
    which,
):
    a, rows = inputs()
    if which is None:
        a = replace(a, value=None)
    else:
        rows = tuple(replace(m, value=None) if m.channels == which else m for m in rows)
    r = module.decompose_fare_spike(a, rows, spike_definition="level_difference")
    assert r.status == "incomplete_measurements" and len(r.measurements) == 8
    assert (
        all(v is None for _, v in r.components)
        and not r.record()["exact_efficiency_verified"]
    )
    assert len(r.missing) == 1 and not r.record()["H21_inference_or_full_B2_adoption"]
    assert r.record()["scientific_budget"] is None


def test_missing_duplicate_or_no_shock_substitution_for_empty_subset_refused():
    a, rows = inputs()
    for bad in [
        rows[:-1],
        rows[:-1] + (rows[0],),
        tuple(a if m.channels == frozenset() else m for m in rows),
    ]:
        with pytest.raises((ValueError, TypeError)):
            module.decompose_fare_spike(a, bad, spike_definition="level_difference")
    with pytest.raises(TypeError, match="no-shock"):
        module.decompose_fare_spike(rows[0], rows, spike_definition="level_difference")


@pytest.mark.parametrize(
    "change",
    [
        {"pair_id": "other"},
        {"cohort_seed": 32},
        {"metric_id": "p90_fare"},
        {"input_unit": "other unit"},
        {"weighting_id": "other weights"},
        {"observation_basis_id": "different support"},
        {"physical_bank_sha256": sha("other bank")},
        {"reveal_utc": "2026-03-08T01:00:00Z"},
        {"baseline_run_id": "another baseline"},
    ],
)
def test_contract_mismatch_never_silently_pairs_other_observations(change):
    a, rows = inputs()
    bad = replace(rows[0], contract=replace(a.contract, **change))
    with pytest.raises(ValueError, match="contracts differ"):
        module.decompose_fare_spike(
            a, (bad,) + rows[1:], spike_definition="level_difference"
        )


def test_batch_keeps_mean_P90_and_unknown_contexts_separate_and_order_independent():
    a, rows = inputs()
    b, other = inputs(c=contract(metric_id="p90_fare"))
    c, unknown = inputs(c=contract(pair_id="unknown"))
    unknown = tuple(
        replace(m, value=None) if m.channels == frozenset("S") else m for m in unknown
    )
    all_rows = (a, *rows, b, *other, c, *unknown)
    out = module.paired_decompositions(all_rows, spike_definition="level_difference")
    reverse = module.paired_decompositions(
        tuple(reversed(all_rows)), spike_definition="level_difference"
    )
    assert [r.record() for r in out] == [r.record() for r in reverse]
    assert (
        len(out) == 3 and sum(r.status == "incomplete_measurements" for r in out) == 1
    )
    assert len([r for r in out if r.baseline.contract.pair_id == "unit-context"]) == 2
    assert module.paired_decompositions([], spike_definition="level_difference") == ()


def test_lossless_json_retains_negative_zero_declared_literal_and_rational_inputs():
    c = contract()
    for value in [
        "-0.0000",
        "-1.123456789012345678901",
        Fraction(-2, 3),
        {"numerator": "-004", "denominator": "-006"},
        None,
    ]:
        m = measurement(c, frozenset("M"), value)
        record = json.loads(json.dumps(m.record()))
        assert module.FareMetricMeasurement.from_record(record).record() == record
    a, rows = inputs()
    r = module.decompose_fare_spike(a, rows, spike_definition="level_difference")
    record = json.loads(json.dumps(r.record()))
    assert record["frozen_residual"] == {"numerator": "7", "denominator": "1"}
    assert record["source_and_channel_semantics_authenticated"] is False
    assert record["declared_hashes_are_not_original_transitive_seals"]


def test_supplied_literal_and_canonical_value_disagreement_is_not_repaired():
    m = measurement(contract(), frozenset("M"), "12.345")
    record = m.record()
    record["value"] = {"numerator": "0", "denominator": "1"}
    with pytest.raises(ValueError, match="differs"):
        module.FareMetricMeasurement.from_record(record)


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        True,
        "NaN",
        "Infinity",
        {"numerator": 1.9, "denominator": 2},
        {"numerator": "1", "denominator": "0"},
    ],
)
def test_invalid_precision_and_nonfinite_values_are_not_coerced_or_truncated(value):
    with pytest.raises((ValueError, TypeError)):
        measurement(contract(), frozenset("M"), value)


def test_72_elapsed_hour_halfopen_window_normalizes_offsets_across_DST():
    from zoneinfo import ZoneInfo

    c = contract()
    ny = ZoneInfo("America/New_York")
    assert c.end_utc - c.reveal_utc == timedelta(hours=72)
    assert c.contains(c.reveal_utc) and c.contains(
        c.end_utc - timedelta(microseconds=1)
    )
    assert not c.contains(c.reveal_utc - timedelta(microseconds=1)) and not c.contains(
        c.end_utc
    )
    assert c.contains(c.reveal_utc.astimezone(ny))
    assert c.end_utc.astimezone(ny).replace(tzinfo=None) - c.reveal_utc.astimezone(
        ny
    ).replace(tzinfo=None) == timedelta(hours=73)
    assert replace(c, reveal_utc=c.reveal_utc.astimezone(ny)) == c
    with pytest.raises(ValueError, match="exactly 72"):
        contract(window_hours=48)
    with pytest.raises(ValueError, match="timezone-aware"):
        contract(reveal_utc=datetime(2026, 3, 8))


def test_no_actor_simulation_RNG_mutation_or_positive_scientific_claim():
    a, rows = inputs()
    before = pickle.dumps((a, rows))
    rng = pickle.dumps(np.random.get_state())
    r = module.decompose_fare_spike(a, rows, spike_definition="level_difference")
    assert before == pickle.dumps((a, rows)) and rng == pickle.dumps(
        np.random.get_state()
    )
    assert not r.record()["H21_inference_or_full_B2_adoption"]
    assert not r.record()["source_and_channel_semantics_authenticated"]
    assert r.record()["scientific_budget"] is None


def test_declared_literals_and_exported_records_have_independent_immutable_ownership():
    literal = {"numerator": "2", "denominator": "3"}
    m = measurement(contract(), frozenset("M"), literal)
    literal["numerator"] = "100"
    assert (
        m.value == Fraction(2, 3)
        and m.record()["declared_value_literal"]["numerator"] == "2"
    )
    with pytest.raises(TypeError):
        m.declared_value_literal["numerator"] = "99"
    record = m.record()
    record["declared_value_literal"]["numerator"] = "999"
    assert m.record()["declared_value_literal"]["numerator"] == "2"
    assert module.FareMetricMeasurement.from_record(m.record()).value == Fraction(2, 3)


def test_empty_metric_residual_diagnostic_is_not_zero_variance_or_equivalence_claim():
    assert result().record()["frozen_declared_metric_difference_is_zero"] is False
    r = result(residual=0)
    assert r.record()["frozen_declared_metric_difference_is_zero"] is True
    assert r.record()["H21_inference_or_full_B2_adoption"] is False


def test_wrong_baseline_id_duplicate_batch_baselines_and_undefined_spike_convention_refused():
    a, rows = inputs()
    with pytest.raises(ValueError, match="common baseline"):
        module.decompose_fare_spike(
            replace(a, run_id="not-the-baseline"),
            rows,
            spike_definition="level_difference",
        )
    with pytest.raises(ValueError, match="ONE separate"):
        module.paired_decompositions((a, a, *rows), spike_definition="level_difference")
    with pytest.raises(ValueError, match="explicit level_difference"):
        module.decompose_fare_spike(a, rows, spike_definition="implicit percentage")


def test_channel_and_hash_declarations_are_exact_not_guessed_or_authenticated():
    c = contract()
    for channels in [("M", "M"), ("X",), "MF"]:
        with pytest.raises((TypeError, ValueError)):
            measurement(c, channels, 1)
    with pytest.raises(ValueError, match="SHA256"):
        contract(physical_bank_sha256="unknown")
    assert result().record()["source_and_channel_semantics_authenticated"] is False
