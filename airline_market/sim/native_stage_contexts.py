"""A.10/A.5/A.14/B0/V9: pure complete native attempt/stage context binding.

Computation inputs and earlier native facts are not received knowledge or a law.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from airline_market.core.hashing import canonical_json, jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.native_information_context_v2 import (
    native_information_context_record,
)
from airline_market.sim.passenger_accounts import _binary, _encode
from airline_market.sim.received_information import _text, _units
from airline_market.sim.reentry_stage_information import RULE as STAGE_RULE

RULE = "complete-native-attempt-and-reached-stage-contexts-v1"
SCOPE = "native-choice-inputs-before-stage-utility-not-received-knowledge"


class NativeStageContextLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeStageContexts:
    analyst_context: object
    periods: tuple
    people: tuple
    definition: str
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        for name in ("analyst_context", "periods", "people"):
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))


def _same(a, b, message):
    if canonical_json(_encode(jsonable(a))) != canonical_json(_encode(jsonable(b))):
        raise ValueError(message)


def _fields(row, names, message):
    if not isinstance(row, Mapping) or set(row) != set(names.split()):
        raise ValueError(message)


def _fare(value):
    """Validate the whole supplied raw binary64 envelope, without coercion/repair."""
    _fields(value, "binary64_bits float_hex repr exact", "full fare envelope required")
    bits = value["binary64_bits"]
    if not isinstance(bits, str) or len(bits) != 16:
        raise ValueError("binary64 fare bits required")
    try:
        number = struct.unpack(">d", bytes.fromhex(bits))[0]
    except (ValueError, struct.error) as exc:
        raise ValueError("binary64 fare bits required") from exc
    _same(value, _binary(number), "fare envelope fields differ from supplied bits")


def _source(context, stage):
    """Full shared support, not a source-time/scenario truth authentication."""
    _fields(
        stage,
        "plan_id rule row_limit source_context periods pending_inputs pending_results actual_received_information_type_probability_or_utility_authenticated native_choice_law_actor_observations_metrics_defaults_changed native_reentry_stage_inputs_captured executor_revision_scope lifecycle_utility_CS_or_payoff scientific_budget",
        "complete stage source contract required",
    )
    if stage.get("rule") != STAGE_RULE:
        raise ValueError("current complete stage source rule required")
    if type(stage["row_limit"]) is not int or stage["row_limit"] < 1:
        raise ValueError("positive supplied source infrastructure limit required")
    if (
        stage["actual_received_information_type_probability_or_utility_authenticated"]
        is not False
        or stage["native_choice_law_actor_observations_metrics_defaults_changed"]
        is not False
        or stage["lifecycle_utility_CS_or_payoff"] is not None
        or stage["scientific_budget"] is not None
    ):
        raise ValueError(
            "no received knowledge/utility/default adoption in source contract"
        )
    if (
        stage.get("pending_inputs") is not None
        or stage.get("pending_results") is not None
    ):
        raise ValueError("complete closed stage source required; no pending salvage")
    source = context["supplied_source"]["analyst_context"]
    actual = stage["source_context"]
    _fields(
        actual,
        "native_spec spec_hash scenario dated_rebooking dated_menu dated_reserve cap_policy trigger_rule mechanical_benchmark ordinary_information own_information proof_information presentation_calls final_own_histories own_tickets own_keys",
        "full stage source contracts and original support required",
    )
    for name in ("native_spec", "spec_hash"):
        _same(source[name], actual[name], "same full native model required")
    _same(
        source["clock"], actual["scenario"]["clock_utc"], "same full UTC clock required"
    )
    _same(
        source["periods"],
        actual["ordinary_information"],
        "ALL ordinary source rows required",
    )
    for a, b in (
        ("own_periods", "own_information"),
        ("proof_periods", "proof_information"),
        ("presentation_calls", "presentation_calls"),
    ):
        _same(context[a], actual[b], "ALL matching own/proof/call support required")
    histories = context["final_prospective_histories"]
    for name in ("final_own_histories", "own_tickets", "own_keys"):
        if set(actual[name]) != {str(h[0]) for h in histories}:
            raise ValueError("ALL original root histories/tickets/keys required")
    for root, facts, tickets, key in histories:
        for name, expected in (
            ("final_own_histories", facts),
            ("own_tickets", tickets),
            ("own_keys", key),
        ):
            _same(
                actual[name][str(root)],
                expected,
                "same complete original root support required",
            )
    n = source["completed_source_periods"]
    if not (len(stage["periods"]) == len(source["periods"]) == n):
        raise ValueError("ALL complete periods including empty periods required")
    if stage["native_reentry_stage_inputs_captured"] is not any(
        p["inputs"] for p in stage["periods"]
    ):
        raise ValueError("actual completed stage-row flag must match full support")
    return source, actual


def _input(row, result, *, index, ordinal, attempt, source, earlier):
    _fields(
        row,
        "analyst_root analyst_original_key period stage_ordinal route prices offered view",
        "complete reached input schema required",
    )
    _fields(
        result,
        "input_index selected_product outside",
        "complete stage result schema required",
    )
    t, own = attempt.view["native_period"], attempt.view["earlier_own"]
    _same(
        (
            row["analyst_root"],
            row["analyst_original_key"],
            row["period"],
            row["stage_ordinal"],
        ),
        (attempt.analyst_root, attempt.analyst_original_key, t, ordinal),
        "same original person/attempt/stage coordinate required",
    )
    if row["route"] not in {"base", "rescue", "reserve"}:
        raise ValueError("declared native executor route required")
    if type(result["input_index"]) is not int or result["input_index"] != index:
        raise ValueError("same ordered stage result index required")
    if type(result["outside"]) is not bool:
        raise ValueError("explicit stage outside flag required")
    products = source["native_spec"]["products"]
    prices, offered = row["prices"], row["offered"]
    if len(prices) != len(products) or len(offered) != len(products):
        raise ValueError("complete global price and offer arrays required")
    for fare in prices:
        _fare(fare)
    if any(type(flag) is not bool for flag in offered):
        raise ValueError("explicit boolean final stage offers required")
    stage = "own" if ordinal == 0 else "interline"
    market = own["original_market"]
    columns = tuple(i for i, p in enumerate(products) if p["market"] == market)
    for j in columns:
        same_carrier = products[j]["carrier"] == own["original_carrier"]
        if offered[j] and same_carrier != (stage == "own"):
            raise ValueError("stage offers must match original carrier/market")
    expected_view = {
        "native_period": t,
        "compiled_opportunity_utc": source["clock"][t],
        "market_id": market,
        "stage": stage,
        "stage_ordinal": ordinal,
        "earlier_within_attempt_results": earlier,
        "offers": tuple(
            {"product_id": products[j]["id"], "fare": prices[j], "offered": offered[j]}
            for j in columns
        ),
        "outside_available": True,
        "scope": SCOPE,
    }
    _same(
        row["view"],
        expected_view,
        "exact current-market input projection required; no outcome/private fields",
    )
    chosen = result["selected_product"]
    if result["outside"]:
        if chosen is not None:
            raise ValueError("outside stage has no selected physical product")
    elif type(chosen) is not int or chosen not in columns or not offered[chosen]:
        raise ValueError("selected product must be in the SAME final stage offer")
    return expected_view


def bind_native_stage_contexts(native_contexts, stage_record, *, definition, row_limit):
    """Bind NEW supplied complete exports; no producer/native/likelihood call.

    Full model/clock/history compatibility cannot prove source-time or reception.
    """
    _text(definition)
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive all-support structural limit required")
    native_record = native_information_context_record(native_contexts)
    stage = _encode(jsonable(stage_record))
    if not isinstance(stage, Mapping):
        raise TypeError("complete supplied stage record required")
    supplied = {"native_contexts": native_record, "stage_record": stage}
    if _units(supplied) > row_limit:
        raise NativeStageContextLimit(
            "ALL supplied source support exceeds limit; no partial result"
        )
    panel = native_contexts.native
    context = _encode(jsonable(panel.analyst_context))
    source, actual = _source(context, stage)
    reserve = actual["dated_reserve"]
    route = (
        "reserve"
        if reserve is not None and Decimal(reserve["share"]["decimal"]) > 0
        else "rescue"
        if actual["dated_menu"] is not None and actual["dated_rebooking"] is not None
        else "base"
    )
    n = source["completed_source_periods"]
    if len(panel.periods) != n:
        raise ValueError("complete native context prefix required")
    by_key = {
        tuple(r["analyst_person"]["key"]): r
        for p in source["periods"]
        for r in p["rows"]
    }
    full = {h[0]: h[1] for h in context["final_prospective_histories"]}
    periods, coordinates = [], {key: [] for key in by_key}
    native_coordinates = {key: [] for key in by_key}
    for t, (period, saved) in enumerate(zip(panel.periods, stage["periods"])):
        _fields(saved, "period inputs results", "complete stage period schema required")
        if (
            type(saved["period"]) is not int
            or saved["period"] != t
            or period.period != t
        ):
            raise ValueError("complete consecutive native periods required")
        inputs, results = saved["inputs"], saved["results"]
        if len(inputs) != len(results):
            raise ValueError("every reached input needs its separate actual result")
        own_rows = actual["own_information"][t]["reentry_rows"]
        proof_rows = actual["proof_information"][t]["reentry_rows"]
        if not len(period.reentry_contexts) == len(own_rows) == len(proof_rows):
            raise ValueError("every earlier own/proof attempt context required")
        index, joined = 0, []
        for attempt_index, (attempt, own_row, proof_row) in enumerate(
            zip(period.reentry_contexts, own_rows, proof_rows)
        ):
            key, root = attempt.analyst_original_key, attempt.analyst_root
            if (
                key not in by_key
                or root != by_key[key]["analyst_person"]["original_record"]
            ):
                raise ValueError("same original ordinary person/root required")
            for row in (own_row, proof_row):
                _same(
                    (row["analyst_root"], row["analyst_original_key"]),
                    (root, key),
                    "same ordered own/proof person required",
                )
            expected_attempt = {
                "kind": "native_reentry_before_marks",
                "native_period": t,
                "compiled_opportunity_utc": source["clock"][t],
                "original_ordinary_menu": by_key[key]["menu_view"],
                "earlier_own": own_row["view"],
                "earlier_proof": proof_row["view"],
                "current_effective_reentry_menu": None,
                "known_private_type": None,
                "received_disclosures_or_complete_knowledge": None,
                "scope": "captured-native-streams-only-no-cross-stream-instant-order",
            }
            _same(
                attempt.view,
                expected_attempt,
                "earlier context must equal SAME captured own/proof prefix",
            )
            _same(
                attempt.person_decision,
                len(native_coordinates[key]),
                "original-person native decision ordinal differs",
            )
            native_coordinates[key].append((t, "reentry", attempt_index))
            fact_index = len(own_row["view"]["own_facts"])
            outcome = full[root][fact_index]
            if outcome["native_period"] != t or not outcome["kind"].startswith(
                "native_reentry_"
            ):
                raise ValueError(
                    "actual attempt outcome must follow earlier own prefix"
                )
            count = 0
            while index < len(inputs) and inputs[index]["analyst_root"] == root:
                if count >= 2 or (
                    count == 1
                    and (
                        not results[index - 1]["outside"]
                        or own_row["view"]["interline_enabled"] is not True
                    )
                ):
                    raise ValueError(
                        "interline only after actual own outside and enabled contract"
                    )
                earlier = () if count == 0 else ({"stage": "own", "outside": True},)
                menu = _input(
                    inputs[index],
                    results[index],
                    index=index,
                    ordinal=count,
                    attempt=attempt,
                    source=source,
                    earlier=earlier,
                )
                _same(
                    inputs[index]["route"],
                    route,
                    "stage route differs from complete source contract",
                )
                view = {
                    "earlier_native_attempt_context": expected_attempt,
                    "current_native_stage_choice_inputs": menu,
                    "received_information_known_type_or_law": None,
                    "scope": "native-computation-context-not-authenticated-received-information",
                }
                joined.append(
                    {
                        "analyst_original_key": key,
                        "analyst_root": root,
                        "person_decision": attempt.person_decision,
                        "stage_ordinal": count,
                        "view": view,
                    }
                )
                coordinates[key].append((t, "reentry_stage", len(joined) - 1))
                index += 1
                count += 1
            expected_count = (
                2
                if count
                and results[index - count]["outside"]
                and own_row["view"]["interline_enabled"]
                else 1
            )
            if count != expected_count:
                raise ValueError("ALL and ONLY reached own/interline stages required")
            final = results[index - 1]
            product = (
                None
                if final["outside"]
                else source["native_spec"]["products"][final["selected_product"]]["id"]
            )
            if (product, final["outside"]) != (
                outcome["product_id"],
                outcome["kind"] == "native_reentry_outside",
            ):
                raise ValueError(
                    "final reached stage must match prospective native attempt outcome"
                )
            if product is not None:
                _same(
                    outcome["stage"],
                    inputs[index - 1]["view"]["stage"],
                    "final purchase stage differs",
                )
                _same(
                    outcome["fare"],
                    inputs[index - 1]["prices"][final["selected_product"]],
                    "actual purchase fare differs from SAME stage",
                )
        if index != len(inputs):
            raise ValueError("no extra or reordered stage-only attempts permitted")
        original = source["periods"][t]["rows"]
        if len(period.ordinary_contexts) != len(original):
            raise ValueError("ALL original ordinary/outside rows required")
        for ordinary_index, (ordinary, raw) in enumerate(
            zip(period.ordinary_contexts, original)
        ):
            key, root = (
                tuple(raw["analyst_person"]["key"]),
                raw["analyst_person"]["original_record"],
            )
            expected = {
                "kind": "native_ordinary_before_taste_utility_choice",
                "native_period": t,
                "compiled_opportunity_utc": source["clock"][t],
                "ordinary_menu": raw["menu_view"],
                "known_private_type": None,
                "received_disclosures_or_complete_knowledge": None,
            }
            _same(
                (
                    ordinary.analyst_original_key,
                    ordinary.analyst_root,
                    ordinary.person_decision,
                    ordinary.view,
                ),
                (key, root, 0, expected),
                "ALL original ordinary contexts must match supplied source",
            )
            coordinates[key].append((t, "ordinary", ordinary_index))
            native_coordinates[key].append((t, "ordinary", ordinary_index))
        periods.append(
            {
                "period": t,
                "ordinary_contexts": _encode(jsonable(period.ordinary_contexts)),
                "reentry_stage_contexts": tuple(joined),
            }
        )
    expected_parent_people = tuple(
        {
            "analyst_original_key": key,
            "analyst_root": by_key[key]["analyst_person"]["original_record"],
            "context_coordinates": tuple(coords),
        }
        for key, coords in native_coordinates.items()
    )
    _same(
        panel.people,
        expected_parent_people,
        "ALL original parent people and native contexts required",
    )
    people = tuple(
        {
            "analyst_original_key": key,
            "analyst_root": by_key[key]["analyst_person"]["original_record"],
            "stage_context_coordinates": tuple(coords),
        }
        for key, coords in coordinates.items()
    )
    result = NativeStageContexts(
        supplied, tuple(periods), people, definition, row_limit
    )
    if _units(native_stage_context_record(result)) > row_limit:
        raise NativeStageContextLimit(
            "ALL supplied source and proposed full output exceeds limit; no partial result"
        )
    return result


def native_stage_context_record(panel):
    """Owned complete ANALYST export; invokes no source/native/probability producer."""
    if type(panel) is not NativeStageContexts or panel.rule != RULE:
        raise TypeError("complete native stage context result required")
    return {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "result": _encode(jsonable(panel)),
        "native_sources_stage_reach_and_outcome_compatibility_checked": True,
        "actual_received_information_type_law_preferences_or_utility_authenticated": False,
        "native_behavior_actors_observations_metrics_defaults_changed": False,
        "lifecycle_utility_CS_service_or_payoff": None,
        "scientific_budget": None,
    }
