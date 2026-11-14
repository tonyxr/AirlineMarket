"""A.10/A.5/B0/V9: prospective identity for ALL actual ordinary requests.

Capture existing native log rows, including outside choices, once per original
request. Re-entry searches stay on the original booked account, never new people.
"""

import copy
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_demand import DatedDemandMarketEngine
from airline_market.policy.dated_order_admission import DatedOrderBenchmarkEngine
from airline_market.sim.passenger_accounts import OriginalPassengerAccounts, _encode

RULE = "all-ordinary-potential-requests-v1"


@dataclass(frozen=True)
class OrdinaryPerson:
    key: tuple
    native_rank: int
    segment: str
    taste_class: int
    request_market: str
    native_log_index: int
    original_record: int | None
    native_row: object


@dataclass(frozen=True)
class OrdinaryPeriod:
    period: int
    log_start: int
    log_stop: int
    rows: tuple


@dataclass(frozen=True)
class OrdinaryCohortSnapshot:
    native: object
    periods: tuple
    pending: object
    capture_rule: str = RULE


class _OrdinaryCohort:
    """Cooperative sidecar around the existing genuine-request lineage wrapper."""

    def __init__(self, *args, **kwargs):
        self._ordinary_periods = []
        self._pending_ordinary = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._ordinary_periods = []
        self._pending_ordinary = None
        return super().reset(seed, tape=tape)

    def _track_requests(self, requests):
        for request in super()._track_requests(requests):
            period = self.world.t
            ticket = self.world.next_record_id
            index = len(self.world.request_log)
            yield request
            if len(self.world.request_log) != index + 1:
                raise RuntimeError(
                    "one existing native log row per ordinary request required"
                )
            row = self.world.request_log[-1]
            if (row["period"], row["market"], row["segment"]) != (
                period,
                request.market,
                request.segment,
            ):
                raise RuntimeError(
                    "native ordinary row and genuine request identity differ"
                )
            chosen = row["chosen"] is not None
            if self.world.next_record_id - ticket != int(chosen):
                raise RuntimeError("native ordinary purchase identity differs")
            self._pending_ordinary.append(
                OrdinaryPerson(
                    (period, request.gumbel_row, request.market),
                    request.rank,
                    self.compiled.spec.segments[request.segment].id,
                    request.taste_class,
                    self.compiled.spec.markets[request.market].id,
                    index,
                    ticket if chosen else None,
                    deep_freeze(row),
                )
            )

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError(
                "ordinary cohort requires genuine native requests and choices"
            )
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal native opportunity required")
        period = self.world.t
        before = len(self.world.request_log)
        self._pending_ordinary = []
        result = super().step(actions)
        rows = tuple(self._pending_ordinary)
        stop = len(self.world.request_log)
        if (
            self.world.t != period + 1
            or tuple(r.native_log_index for r in rows) != tuple(range(before, stop))
            or len({r.key for r in rows}) != len(rows)
        ):
            raise RuntimeError(
                "complete ordinary cohort and native closed period differ"
            )
        for row in rows:
            if row.original_record is not None:
                coord = self._physical_coordinates[row.original_record]
                if coord.stream_key != row.key or coord.native_rank != row.native_rank:
                    raise RuntimeError(
                        "booked original lineage and ordinary cohort differ"
                    )
        self._ordinary_periods.append(OrdinaryPeriod(period, before, stop, rows))
        self._pending_ordinary = None
        return result

    def ordinary_periods(self):
        if self._pending_ordinary is not None:
            raise ValueError(
                "incomplete ordinary capture; no failed-run salvage or partial cohort"
            )
        return tuple(self._ordinary_periods)

    def snapshot(self):
        return OrdinaryCohortSnapshot(
            super().snapshot(),
            tuple(self._ordinary_periods),
            None if self._pending_ordinary is None else tuple(self._pending_ordinary),
        )

    def restore(self, snapshot):
        if type(snapshot) is not OrdinaryCohortSnapshot:
            raise TypeError("ordinary cohort combined snapshot required")
        if snapshot.capture_rule != RULE:
            raise ValueError("snapshot ordinary capture rule differs")
        super().restore(snapshot.native)
        self._ordinary_periods = list(snapshot.periods)
        self._pending_ordinary = (
            None if snapshot.pending is None else list(snapshot.pending)
        )

    def branch(self):
        kwargs = {
            "ordinary_priority": self.ordinary_priority,
            "trigger_rule": self.trigger.rule if self.trigger else None,
            "mechanical_benchmark": self.trigger.benchmark if self.trigger else None,
            "dated_rebooking": self.dated_rebooking,
            "dated_menu": self.dated_menu,
            "dated_reserve": self.dated_reserve,
            "cap_policy": self.cap_policy,
            "check_invariants": self.check_invariants,
        }
        if isinstance(self, DatedOrderBenchmarkEngine):
            kwargs["order_policy"] = self.order_policy
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            **kwargs,
        )
        other.restore(self.snapshot())
        return other


class OrdinaryCohortMarketEngine(_OrdinaryCohort, DatedDemandMarketEngine):
    """Dated chronological/random-priority native paths, plus analyst cohort rows."""


class OrdinaryCohortOrderEngine(_OrdinaryCohort, DatedOrderBenchmarkEngine):
    """Opt-in dated order benchmark with the same ordinary cohort capture rule."""


def ordinary_person_panel(periods, accounts: OriginalPassengerAccounts):
    """Join NEW supplied prefix rows/accounts; do not re-run either producer.

    Every ordinary no-purchase gets its physical request key but no invented
    ticket. Existing ordinary CS stays on its original opportunity; no re-entry
    utility is added. Structural joins do not authenticate a common source/bank.
    """
    if type(accounts) is not OriginalPassengerAccounts:
        raise TypeError("OriginalPassengerAccounts required")
    periods = tuple(periods)
    data = accounts.data
    if any(type(p) is not OrdinaryPeriod for p in periods) or tuple(
        p.period for p in periods
    ) != tuple(range(data["completed_native_periods"])):
        raise ValueError(
            "every completed ordinary period, including empty periods, required"
        )
    flat = []
    end = 0
    for period in periods:
        if (
            period.log_start != end
            or period.log_stop < end
            or tuple(r.native_log_index for r in period.rows)
            != tuple(range(end, period.log_stop))
        ):
            raise ValueError(
                "full contiguous original native request log support required"
            )
        if any(
            type(r) is not OrdinaryPerson
            or r.key[0] != period.period
            or r.native_row["period"] != period.period
            for r in period.rows
        ):
            raise ValueError("ordinary native period/row correspondence required")
        flat.extend(period.rows)
        end = period.log_stop
    if len({r.key for r in flat}) != len(flat):
        raise ValueError("distinct original ordinary physical keys required")
    lineages = {r["original_record"]: r for r in data["lineages"]}
    booked = [r for r in flat if r.original_record is not None]
    if len(booked) != len(lineages) or {r.original_record for r in booked} != set(
        lineages
    ):
        raise ValueError(
            "all originally booked ordinary rows must join accounts exactly once"
        )
    people = []
    outside = 0
    for row in sorted(flat, key=lambda r: r.key):
        if row.native_row["market"] != row.key[2]:
            raise ValueError("ordinary request market and native row differ")
        account = None if row.original_record is None else lineages[row.original_record]
        if (row.native_row["chosen"] is None) != (account is None):
            raise ValueError("ordinary outside/purchase and account support differ")
        if account is not None and (
            account["potential_coordinate"],
            account["native_original_rank"],
            account["segment"],
            account["request_market"],
        ) != (row.key, row.native_rank, row.segment, row.request_market):
            raise ValueError("original booked request and account identity differ")
        outside += account is None
        people.append(
            {
                "potential_coordinate": row.key,
                "native_rank": row.native_rank,
                "segment": row.segment,
                "taste_class": row.taste_class,
                "request_market": row.request_market,
                "native_log_index": row.native_log_index,
                "original_record": row.original_record,
                "original_ordinary_log": _encode(row.native_row),
                "booked_passenger_account": account,
                "lifecycle_consumer_surplus": None,
                "delivered_service_or_causal_denial": None,
            }
        )
    if outside != data["ordinary_outside_choices_in_prefix"]:
        raise ValueError("outside ordinary prefix count and account context differ")
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "capture_rule": RULE,
            "population": "ALL actual thinned ordinary requests in the supplied completed native prefix, including outside choices",
            "completed_native_periods": len(periods),
            "period_request_counts": tuple(len(p.rows) for p in periods),
            "ordinary_original_requests": len(people),
            "originally_booked": len(booked),
            "ordinary_outside": outside,
            "persons": tuple(people),
            "reentry_searches_are_additional_people": False,
            "original_ordinary_CS_reclassified_after_stranding": False,
            "same_producer_bank_original_capture_authenticated": False,
            "empirical_exposure_or_travellers_inferred": False,
            "total_lifecycle_surplus": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_B4_welfare_adopted": False,
            "scientific_budget": None,
        }
    )


def ordinary_panel_record(panel):
    """Owned JSON export with lossless binary64 native-log/account envelopes."""
    return jsonable(panel)
