"""A.10/A.5/A.14/B0/V9/B4: complete native postings and declared role laws.

Pure supplied-source association. No native probability, received knowledge,
private-vector awareness, preference or lifecycle consumer surplus is inferred.
"""

import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.sim import action_kernel_utility as utility
from airline_market.sim import context_compatibility as context
from airline_market.sim import period_posting as posting
from airline_market.sim import role_law_context as role
from airline_market.sim.native_stage_contexts import _fare
from airline_market.sim.shared_mark_attempt import _encoded, _owned

RULE = "complete-posting-role-context-association-v1"


@dataclass(frozen=True)
class PostingRoleBinding:
    posting_source: posting.PeriodPostingSource
    attached: role.RoleContextResult
    input_digest: str
    output_digest: str
    required_nodes: int
    support_limit: int
    native_G: None = None
    native_probability_knowledge_preferences_or_CS_authenticated: bool = False
    rule: str = RULE

    def export(self):
        return posting_role_record(self)


def _hash(value):
    return utility._digest(_encoded(value))


def _fields(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names.split()):
        raise ValueError("complete current posting schema required")


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise TypeError("exact nonnegative integer source stamp required")


def _source(value):
    if type(value) is not posting.PeriodPostingSource or value.rule != posting.RULE:
        raise TypeError("complete typed current posting source required")
    # Reject opaque mutable objects instead of relying on serializer coercion.
    owned = posting.PeriodPostingSource(_owned(value.body), value.digest, value.rule)
    body = owned.body
    _fields(
        body,
        "rule contract required_support parent_source periods posting_calls native_probability_received_knowledge_preferences_or_CS_authenticated lifecycle_utility_CS scientific_budget",
    )
    c = body["contract"]
    _fields(
        c,
        "rule parent_limit posting_call_limit support_limit products horizon clock causal_programme_hash",
    )
    for name in ("parent_limit", "posting_call_limit", "support_limit", "horizon"):
        _integer(c[name], 1)
    for name in ("required_support", "posting_calls"):
        _integer(body[name])
    if c["rule"] != posting.RULE or len(c["clock"]) != c["horizon"] + 1:
        raise ValueError("complete current posting rule and T+1 clock required")
    if (
        not c["products"]
        or len(set(c["products"])) != len(c["products"])
        or any(type(p) is not str or not p for p in c["products"])
    ):
        raise ValueError("complete distinct physical-product IDs required")
    for row in body["periods"]:
        _fields(
            row,
            "period compiled_opportunity_utc postings input_links native_closed_world_t",
        )
        _integer(row["period"])
        _integer(row["native_closed_world_t"])
        for post in row["postings"]:
            _fields(
                post,
                "native_world_t_at_post prices native_cap_receipt earlier_global_journal earlier_global_event_count",
            )
            _integer(post["native_world_t_at_post"])
            _integer(post["earlier_global_event_count"])
            cap = post["native_cap_receipt"]
            _fields(
                cap,
                "plan_id period revision products any_price_constraint_binding excess_demand_measured lottery_group_adopted native_rationing_adopted",
            )
            for name in (
                "any_price_constraint_binding",
                "excess_demand_measured",
                "lottery_group_adopted",
                "native_rationing_adopted",
            ):
                if type(cap[name]) is not bool:
                    raise TypeError("exact Boolean native receipt flags required")
            for product in cap["products"]:
                _fields(
                    product,
                    "product native_proposal static_ceiling after_static trigger_ceiling executed static_clipped trigger_clipped price_constraint_binding",
                )
                for name in (
                    "static_clipped",
                    "trigger_clipped",
                    "price_constraint_binding",
                ):
                    if type(product[name]) is not bool:
                        raise TypeError("exact Boolean native product flags required")
            for name in ("period", "revision"):
                _integer(post["native_cap_receipt"][name], 0 if name == "period" else 1)
        for link in row["input_links"]:
            _fields(
                link,
                "posting_revision kind parent_input_index original_key root global_input_event_count",
            )
            for name in (
                "posting_revision",
                "parent_input_index",
                "global_input_event_count",
            ):
                _integer(link[name], 1 if name == "posting_revision" else 0)
            context._key(link["original_key"])
            if link["root"] is not None:
                _integer(link["root"])
    # Frozen PURE stored-source validation, never a producer/engine forward.
    posting.period_posting_record(owned)
    return owned


def _postings(source):
    products = source.body["contract"]["products"]
    result = []
    for row in source.body["periods"]:
        prices = row["postings"][-1]["prices"]
        if len(prices) != len(products):
            raise ValueError("ALL actual final physical-product prices required")
        values = []
        for product, envelope in zip(products, prices):
            _fare(envelope)
            value = struct.unpack(">d", bytes.fromhex(envelope["binary64_bits"]))[0]
            if not role._same(_encoded(value), envelope):
                raise ValueError("lossless actual binary64 posting required")
            values.append((product, value))
        result.append(tuple(values))
    return tuple(result)


def _prepare(registration, source, periods, selected, manifests):
    source = _source(source)
    reg = role._validated_registration(registration)
    parent, periods, selected = context._inputs(
        source.body["parent_source"], periods, selected
    )
    if not role._same(parent["periods"], periods):
        raise ValueError("SAME complete source and ALL typed native periods required")
    if tuple(source.body["contract"]["clock"]) != tuple(
        t.isoformat() for t in periods[-1].journal.clock
    ):
        raise ValueError("posting clock must match SAME full native journal clock")
    postings, manifests = _postings(source), role._manifests(manifests)
    return source, reg, parent, periods, selected, postings, manifests


def _input_payload(source, reg, periods, selected, manifests, limit):
    return source, reg, periods, selected, manifests, limit, RULE


def _result_payload(value):
    return (
        value.posting_source,
        value.attached,
        value.input_digest,
        value.required_nodes,
        value.support_limit,
        value.native_G,
        value.native_probability_knowledge_preferences_or_CS_authenticated,
        RULE,
    )


def _export_shape(value):
    """Widen nullable numerical slots for the lossless Fraction export.

    The parent shape counts scalar fractions. This wrapper's counter sees their
    numerator/denominator records. Widening every NULL is conservative, including
    known immutable input NULLs; actual source values are never changed.
    """
    value = _encoded(value)

    def widen(item):
        if item is None:
            return _encoded(Fraction(0))
        if isinstance(item, Mapping):
            return {key: widen(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return tuple(widen(child) for child in item)
        return item

    return widen(value)


def _required(source, reg, parent, periods, selected, postings, manifests, limit):
    # Adapt the frozen PURE shape, not a native business algorithm. Every future
    # conditional row is retained, including zero/unreachable inputs. Full receipt
    # is a conservative substitute for the actual narrow earlier view; the full
    # nested action-kernel shape also covers the upstream-NULL result.
    inner = role.RoleContextResult(
        reg,
        context._shape(parent, periods, selected, role._plan(reg)),
        postings,
        manifests,
        Fraction(0),
        Fraction(0),
        reg.raw.subjective.prior,
        reg.raw.reference.prior,
        utility._shape(reg.raw),
        "digest",
        0,
        limit,
    )
    outer = PostingRoleBinding(source, inner, "digest", "digest", 0, limit)
    return context._count(
        {
            "inputs": _input_payload(source, reg, periods, selected, manifests, limit),
            "derived_postings": postings,
            "output": _export_shape(outer),
        }
    )


def _links(source, reg, parent, periods, selected, postings, manifests):
    role._source_agreement(reg, parent, periods, selected, postings, manifests)
    if (
        source.body[
            "native_probability_received_knowledge_preferences_or_CS_authenticated"
        ]
        is not False
    ):
        raise ValueError("unprovided native knowledge/law/preferences remain unadopted")


def _attached_agreement(attached, reg, parent, periods, selected, prices, manifests):
    if type(attached) is not role.RoleContextResult or not all(
        role._same(actual, expected)
        for actual, expected in (
            (attached.registration, reg),
            (attached.context.source_record, parent),
            (attached.context.periods, periods),
            (attached.context.selected_input, selected),
            (attached.actual_postings, prices),
            (attached.actual_manifests, manifests),
        )
    ):
        raise ValueError("SAME full source/registration/context/postings required")


def bind_posting_role_context(
    registration,
    posting_source,
    periods,
    selected_input,
    *,
    actual_manifests,
    support_limit,
):
    """Own complete source/context; delegate ONE declared role attachment.

    Actual regular prices are decoded from the complete stored binary64 record.
    No engine, source producer, actor, RNG, native utility or saved loader is run.
    """
    source, reg, parent, periods, selected, prices, manifests = _prepare(
        registration, posting_source, periods, selected_input, actual_manifests
    )
    needed = _required(
        source, reg, parent, periods, selected, prices, manifests, support_limit
    )
    context._limit(support_limit, needed)
    _links(source, reg, parent, periods, selected, prices, manifests)
    attached = role.attach_role_context(
        reg,
        parent,
        periods,
        selected,
        actual_postings=prices,
        actual_manifests=manifests,
        support_limit=support_limit,
    )
    _attached_agreement(attached, reg, parent, periods, selected, prices, manifests)
    value = PostingRoleBinding(
        source,
        attached,
        _hash(_input_payload(source, reg, periods, selected, manifests, support_limit)),
        "digest",
        needed,
        support_limit,
    )
    if (
        context._count(
            {
                "inputs": _input_payload(
                    source, reg, periods, selected, manifests, support_limit
                ),
                "derived_postings": prices,
                "output": value,
            }
        )
        > needed
    ):
        raise ValueError("actual complete binding exceeds proposed structural support")
    return replace(value, output_digest=_hash(_result_payload(value)))


def _validated(value):
    if type(value) is not PostingRoleBinding or value.rule != RULE:
        raise TypeError("current typed complete posting-role binding required")
    if (
        value.native_G is not None
        or value.native_probability_knowledge_preferences_or_CS_authenticated
        is not False
    ):
        raise ValueError("native G/law/knowledge/preferences/CS stay NULL/false")
    attached = role._validated_result(value.attached)
    source, reg, parent, periods, selected, prices, manifests = _prepare(
        attached.registration,
        value.posting_source,
        attached.context.periods,
        attached.context.selected_input,
        attached.actual_manifests,
    )
    needed = _required(
        source, reg, parent, periods, selected, prices, manifests, value.support_limit
    )
    if type(value.required_nodes) is not int or value.required_nodes != needed:
        raise ValueError("complete recomputed input/proposed-output support differs")
    context._limit(value.support_limit, needed)
    if attached.support_limit != value.support_limit:
        raise ValueError("same full nested attachment support contract required")
    _links(source, reg, parent, periods, selected, prices, manifests)
    _attached_agreement(attached, reg, parent, periods, selected, prices, manifests)
    if value.input_digest != _hash(
        _input_payload(source, reg, periods, selected, manifests, value.support_limit)
    ) or value.output_digest != _hash(_result_payload(value)):
        raise ValueError("complete stored source/registration/result integrity differs")
    return value


def posting_role_record(value):
    """Owned full ANALYST export, without classifier/solver/source execution."""
    return _encoded(_validated(value))


def earlier_posting_role_view(value):
    """ONLY exact earlier native H; full sources/laws/current results excluded."""
    value = _validated(value)
    return _encoded(value.attached.context.earlier_view)
