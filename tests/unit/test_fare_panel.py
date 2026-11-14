"""B2/H2.1: NEW native public-mask exports, byte-linked metrics and frozen arithmetic bridge."""

import hashlib
import importlib.util
import json
import os
import pickle
import struct
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.eval.spike_decomposition import paired_decompositions
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from tests.unit import test_rationing_native as native

if os.environ.get("B2_PANEL_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_fare_panel", os.environ["B2_PANEL_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.eval import fare_panel as module

BASE = datetime(2026, 3, 7, tzinfo=timezone.utc)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def clock(values=(0, 12, 24, 48, 72, 96)):
    return UtcBookingClock(tuple(BASE + timedelta(hours=x) for x in values))


def engine():
    c = native.fixture().compiled
    e = MarketEngine(c)
    e.reset(
        tape=ExogenousTape.generate(
            c, 31, registry=StreamRegistry(37974, "unit/B2_fare_panel")
        )
    )
    return e


def actions(e, fare=None):
    return (
        native.actions(e)
        if fare is None
        else {
            s.code: PriceAction({j: fare for j in e.compiled.products_of[i]})
            for i, s in enumerate(e.compiled.spec.carriers)
        }
    )


def definition(e, *, weights=None, products=("A_OH0", "B_OH0")):
    return module.FarePanelDefinition(
        products,
        tuple(
            (t, p, "1" if weights is None else weights(t, p))
            for t in range(5)
            for p in products
        ),
        "unit fixed opportunity-product weights",
        "exact_binary64",
        "weighted_inverse_cdf",
    )


def pairing(e, c, d):
    return module.panel_contract(
        e.compiled,
        c,
        d,
        pair_id="unit-panel-pair",
        cohort_seed=31,
        input_unit="unit model regular posting fare",
        physical_bank_sha256=sha(b"unit declared common potential bank"),
        baseline_run_id="unit-baseline",
        reveal_utc=BASE,
        window_hours=72,
    )


def finish(e, fare=None, steps=5):
    for _ in range(steps):
        e.step(actions(e, fare))
    return e


def bind_result(directory, artifact, *, status="completed", bind=True):
    result = {
        "status": status,
        "config_hash": sha(b"unit producer config"),
        "value": None,
    }
    if bind:
        result.update(artifact)
    r = directory / "results.json"
    r.write_text(json.dumps(result, sort_keys=True) + "\n")
    receipt = {
        "status": status,
        "config_hash": result["config_hash"],
        "results_sha256": sha(r.read_bytes()),
    }
    p = directory / "COMPLETE.json"
    p.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return module.SavedFarePanelSource(artifact["posted_panel_path"], r, p)


def source(
    directory,
    e,
    c,
    d,
    contract,
    *,
    channels=None,
    run="unit-baseline",
    status="completed",
    bind=True,
):
    directory.mkdir()
    artifact = module.capture_native_posted_panel(
        e,
        c,
        d,
        contract,
        run_id=run,
        channels=channels,
        destination=directory / "panel.json",
    )
    return bind_result(directory, artifact, status=status, bind=bind)


def read(src, e, c, d, contract, *, channels=None, run="unit-baseline"):
    return module.read_saved_fare_panel(
        src, e.compiled, c, d, contract, run_id=run, channels=channels
    )


def values(metrics):
    return {m.contract.metric_id: m.value for m in metrics.measurements}


def test_actual_native_postings_mean_P90_and_exact_halfopen_window(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "source", e, c, d, p)
    out = read(src, e, c, d, p)
    assert values(out) == {"mean_fare": 250, "p90_fare": 300}
    assert {r["period"] for r in out.rows} == {0, 1, 2, 3} and len(out.rows) == 8
    assert (
        out.total_weight == 8
        and out.unknown_positive_weight == 0
        and out.window_covered
        and not out.reasons
    )
    record = json.loads(json.dumps(out.record()))
    assert record["status"] == "measured_declared_panel"
    assert (
        record["provenance"]["receipt_result_link_verified"]
        and record["provenance"]["results_panel_link_verified"]
    )
    assert (
        not record["provenance"]["original_time_provenance_authenticated"]
        and not record["economic_channels_or_primary_B2_metric_adopted"]
    )


def test_explicit_binary64_precision_and_raw_negative_zero_are_retained(tmp_path):
    e = finish(engine(), fare=0.1)
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    raw = e.trace()
    index = e.compiled.product_index["A_OH0"]
    raw[0] = dict(raw[0], prices=list(raw[0]["prices"]))
    raw[0]["prices"][index] = -0.0
    raw[1] = dict(raw[1], prices=list(raw[1]["prices"]))
    raw[1]["prices"][index] = 0.1
    proxy = SimpleNamespace(
        compiled=e.compiled, tape=e.tape, trace=lambda: raw, terminal=lambda: True
    )
    src = source(tmp_path / "binary", proxy, c, d, p)
    out = read(src, e, c, d, p)
    all_points = [
        module._read_price(row["prices"][prod])
        for row in json.loads((tmp_path / "binary/panel.json").read_text())["rows"][:4]
        for prod in d.public_products
    ]
    assert values(out)["mean_fare"] == sum(all_points) / len(all_points)
    zero = next(
        r for r in out.record()["rows"] if r["period"] == 0 and r["product"] == "A_OH0"
    )
    assert zero["native_repr"] == "-0.0" and zero["binary64_bits"] == "8000000000000000"
    fractional = next(
        r for r in out.rows if r["period"] == 1 and r["product"] == "A_OH0"
    )
    assert (
        module._number(fractional["value"])
        == Fraction.from_float(0.1)
        != Fraction(1, 10)
    )


def test_weighted_inverse_CDF_is_explicit_and_no_interpolation_or_normalization_guess(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e, weights=lambda t, p: "9" if p == "A_OH0" else "1")
    p = pairing(e, c, d)
    out = read(source(tmp_path / "weighted", e, c, d, p), e, c, d, p)
    assert values(out) == {"mean_fare": 210, "p90_fare": 200}
    assert out.total_weight == 40
    with pytest.raises(ValueError, match="explicit supported"):
        replace(d, p90_rule="silently choose linear quantiles")
    with pytest.raises(ValueError, match="exact_binary64"):
        replace(d, price_precision="round to cents")


def test_public_mask_export_has_no_private_events_purchases_unselected_prices_or_actor_calls(
    tmp_path, monkeypatch
):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    before = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())

    def forbidden(*a, **k):
        raise AssertionError("extra actor/simulator/physical draw")

    monkeypatch.setattr(e, "post_actions", forbidden)
    monkeypatch.setattr(e, "step", forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    src = source(tmp_path / "public", e, c, d, p)
    out = read(src, e, c, d, p)
    panel = json.loads(Path(src.panel_path).read_text())
    assert all(
        set(r) == {"period", "prices"} and set(r["prices"]) == set(d.public_products)
        for r in panel["rows"]
    )
    assert not any(
        k in panel
        for k in ("purchases", "events", "capacity", "inventory", "request_log")
    )
    assert before == pickle.dumps(e.snapshot()) and rng == pickle.dumps(
        np.random.get_state()
    )
    assert not out.record()["provenance"]["public_visibility_authenticated"]
    with pytest.raises(FileExistsError):
        module.capture_native_posted_panel(
            e,
            c,
            d,
            p,
            run_id="unit-baseline",
            channels=None,
            destination=src.panel_path,
        )


def test_completed_source_classified_by_status_artifact_even_value_null(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "null-value", e, c, d, p)
    assert json.loads(Path(src.results_path).read_text())["value"] is None
    assert all(m.value is not None for m in read(src, e, c, d, p).measurements)


@pytest.mark.parametrize(
    "status,terminal", [("failed", True), ("time_limit", True), ("running", False)]
)
def test_noncompleted_source_never_gets_full_metrics_even_if_window_prefix_exists(
    tmp_path, status, terminal
):
    e = finish(engine(), steps=5 if terminal else 4)
    c = clock((0, 12, 24, 36, 72, 96))
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "partial", e, c, d, p, status=status, bind=False)
    out = read(src, e, c, d, p)
    assert (
        all(m.value is None for m in out.measurements)
        and "producer_not_completed" in out.reasons
    )
    assert (
        len(out.rows) == 8
        and not out.record()["provenance"]["results_panel_link_verified"]
    )


def test_terminal_native_horizon_shorter_than_72h_keeps_unmeasured_full_window(
    tmp_path,
):
    e = finish(engine())
    c = clock((0, 6, 12, 18, 24, 36))
    d = definition(e)
    p = pairing(e, c, d)
    out = read(source(tmp_path / "short", e, c, d, p), e, c, d, p)
    assert not out.window_covered and "clock_does_not_cover_full_72h" in out.reasons
    assert all(m.value is None for m in out.measurements) and len(out.rows) == 10


def proxy_prices(e, t, product, value):
    raw = [dict(r, prices=list(r["prices"])) for r in e.trace()]
    raw[t]["prices"][e.compiled.product_index[product]] = value
    return SimpleNamespace(
        compiled=e.compiled, tape=e.tape, trace=lambda: raw, terminal=lambda: True
    )


@pytest.mark.parametrize("price", [None, float("inf"), float("-inf"), float("nan")])
def test_unknown_positive_weight_keeps_every_row_and_nulls_metrics(tmp_path, price):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "unknown", proxy_prices(e, 0, "A_OH0", price), c, d, p)
    out = read(src, e, c, d, p)
    assert out.unknown_positive_weight == 1 and len(out.rows) == 8
    assert "unknown_positive_weight_price" in out.reasons and all(
        m.value is None for m in out.measurements
    )


def test_nonfinite_zero_weight_row_is_retained_under_explicit_fixed_definition(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e, weights=lambda t, p: "0" if (t, p) == (0, "A_OH0") else "1")
    p = pairing(e, c, d)
    out = read(
        source(
            tmp_path / "zero-weight", proxy_prices(e, 0, "A_OH0", float("nan")), c, d, p
        ),
        e,
        c,
        d,
        p,
    )
    assert len(out.rows) == 8 and out.unknown_positive_weight == 0
    assert values(out) == {"mean_fare": Fraction(1800, 7), "p90_fare": 300}
    row = next(
        r for r in out.record()["rows"] if r["period"] == 0 and r["product"] == "A_OH0"
    )
    assert row["source_status"] == "nonfinite" and row["weight"] == {
        "numerator": "0",
        "denominator": "1",
    }


def test_zero_total_weight_does_not_become_zero_fare_or_an_empty_positive_result(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e, weights=lambda t, p: "0")
    p = pairing(e, c, d)
    out = read(source(tmp_path / "zero", e, c, d, p), e, c, d, p)
    assert out.total_weight == 0 and "zero_declared_window_weight" in out.reasons
    assert all(m.value is None for m in out.measurements) and len(out.rows) == 8


def test_native_nan_payload_bits_are_retained_exactly(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    bits = "fff8000000000123"
    nan = struct.unpack(">d", bytes.fromhex(bits))[0]
    out = read(
        source(tmp_path / "nan-bits", proxy_prices(e, 0, "A_OH0", nan), c, d, p),
        e,
        c,
        d,
        p,
    )
    row = next(
        r for r in out.record()["rows"] if r["period"] == 0 and r["product"] == "A_OH0"
    )
    assert row["binary64_bits"] == bits and row["value"] is None


def test_actual_panel_bytes_results_path_receipt_and_expected_context_are_checked(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "bound", e, c, d, p)
    panel = Path(src.panel_path)
    original = panel.read_bytes()
    panel.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="actual panel bytes"):
        read(src, e, c, d, p)
    panel.write_bytes(original)
    result = Path(src.results_path)
    raw = result.read_bytes()
    result.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="receipt/results"):
        read(src, e, c, d, p)
    result.write_bytes(raw)
    with pytest.raises(ValueError, match="declarations differ"):
        read(src, e, c, d, replace(p, cohort_seed=32))
    with pytest.raises(ValueError, match="declarations differ"):
        read(src, e, c, d, p, channels=("M",))


def test_actual_weights_and_clock_digest_prevent_same_label_support_substitution(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "basis", e, c, d, p)
    other = definition(e, weights=lambda t, p: "2")
    assert other.weighting_id == d.weighting_id and other.basis_id(
        e.compiled, c
    ) != d.basis_id(e.compiled, c)
    with pytest.raises(ValueError, match="exact declared panel basis"):
        read(src, e, c, other, p)
    with pytest.raises(ValueError):
        read(src, e, clock((0, 13, 24, 48, 72, 96)), d, p)
    with pytest.raises(ValueError, match="EVERY period"):
        replace(d, row_weights=d.row_weights[:-1]).record(e.compiled, c)


def test_new_native_nine_source_bridge_to_frozen_mean_and_P90_decomposition(tmp_path):
    reference = engine()
    c = clock()
    d = definition(reference)
    p = pairing(reference, c, d)
    all_measurements = []
    subsets = [
        None,
        frozenset(),
        frozenset("M"),
        frozenset("F"),
        frozenset("S"),
        frozenset("MF"),
        frozenset("MS"),
        frozenset("FS"),
        frozenset("MFS"),
    ]
    coefficients = {
        frozenset("M"): 10,
        frozenset("F"): -2,
        frozenset("S"): 5,
        frozenset("MF"): 4,
        frozenset("MS"): -6,
        frozenset("FS"): 8,
        frozenset("MFS"): 3,
    }
    for index, k in enumerate(subsets):
        fare = (
            100
            if k is None
            else 107 + sum(v for term, v in coefficients.items() if term <= k)
        )
        e = MarketEngine(reference.compiled)
        e.reset(tape=reference.tape)
        finish(e, fare=fare)
        run = "unit-baseline" if k is None else "unit-case-" + str(index)
        src = source(tmp_path / ("run" + str(index)), e, c, d, p, channels=k, run=run)
        out = read(src, e, c, d, p, channels=k, run=run)
        assert not out.reasons
        all_measurements.extend(out.measurements)
    result = paired_decompositions(
        all_measurements, spike_definition="level_difference"
    )
    assert len(result) == 2
    for out in result:
        assert (
            dict(out.components) == {"M": 10, "F": 5, "S": 7}
            and out.empty_residual == 7
            and out.full_spike == 29
        )
        assert (
            not out.record()["source_and_channel_semantics_authenticated"]
            and not out.record()["H21_inference_or_full_B2_adoption"]
        )


def test_export_touches_no_private_or_unselected_trace_payload(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)

    class Secret:
        def __float__(self):
            raise AssertionError("private/unselected value read")

        def __iter__(self):
            raise AssertionError("private payload iterated")

    rows = []
    for row in e.trace():
        prices = list(row["prices"])
        for j, product in enumerate(e.compiled.spec.products):
            if product.id not in d.public_products:
                prices[j] = Secret()
        rows.append(
            {
                "period": row["period"],
                "prices": prices,
                "purchases": Secret(),
                "events": Secret(),
            }
        )

    class PublicProjectionView:
        compiled = e.compiled
        tape = e.tape

        def trace(self):
            return rows

        def terminal(self):
            return True

        @property
        def world(self):
            raise AssertionError("private world read")

    src = source(tmp_path / "no-private", PublicProjectionView(), c, d, p)
    assert values(read(src, e, c, d, p)) == {"mean_fare": 250, "p90_fare": 300}


def test_missing_native_period_is_retained_unknown_and_never_promoted_from_prefix(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    raw = [r for r in e.trace() if r["period"] != 1]
    proxy = SimpleNamespace(
        compiled=e.compiled, tape=e.tape, trace=lambda: raw, terminal=lambda: True
    )
    out = read(source(tmp_path / "missing-period", proxy, c, d, p), e, c, d, p)
    assert "missing_native_periods" in out.reasons and out.unknown_positive_weight == 2
    assert len(out.rows) == 8 and all(m.value is None for m in out.measurements)
    assert sum(r["source_status"] == "missing_native_period" for r in out.rows) == 2


def test_negative_finite_regular_posting_is_retained_not_winsorized(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    out = read(
        source(tmp_path / "negative", proxy_prices(e, 0, "A_OH0", -25), c, d, p),
        e,
        c,
        d,
        p,
    )
    assert values(out)["mean_fare"] == Fraction(1775, 8)
    row = next(r for r in out.rows if r["period"] == 0 and r["product"] == "A_OH0")
    assert module._number(row["value"]) == -25
    assert not out.record()["economic_channels_or_primary_B2_metric_adopted"]


def test_precision_fields_cannot_be_repaired_by_rebinding_panel_hash(tmp_path):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    src = source(tmp_path / "precision", e, c, d, p)
    path = Path(src.panel_path)
    body = json.loads(path.read_text())
    body["rows"][0]["prices"]["A_OH0"]["value"] = {
        "numerator": "999",
        "denominator": "1",
    }
    path.write_text(json.dumps(body))
    src = bind_result(
        path.parent,
        {"posted_panel_path": str(path), "posted_panel_sha256": sha(path.read_bytes())},
    )
    with pytest.raises(ValueError, match="precision fields disagree"):
        read(src, e, c, d, p)


def test_output_nested_rows_are_independent_and_do_not_mutate_preserved_metrics(
    tmp_path,
):
    e = finish(engine())
    c = clock()
    d = definition(e)
    p = pairing(e, c, d)
    out = read(source(tmp_path / "owned", e, c, d, p), e, c, d, p)
    export = out.record()
    export["rows"][0]["weight"]["numerator"] = "999"
    assert out.record()["rows"][0]["weight"]["numerator"] == "1"
    with pytest.raises(TypeError):
        out.rows[0]["weight"]["numerator"] = "42"
    assert values(out) == {"mean_fare": 250, "p90_fare": 300}
