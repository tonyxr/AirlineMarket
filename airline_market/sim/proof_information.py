"""A.10/A.5/A.14/B0/V9: prospective native itinerary-presentation calls.

Native operator actions and proof status, not actual received disclosure,
complete passenger knowledge, current-stage menus or a probability law.
"""

import hashlib
import json
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.sim.ordinary_information import ordinary_information_view
from airline_market.sim.own_history_information import (
    OwnHistoryInformationMarketEngine,
    own_reentry_information_view,
)
from airline_market.sim.passenger_accounts import _encode

RULE = "prospective-native-itinerary-presentation-before-reentry-marks-v1"


@dataclass(frozen=True)
class NativePresentationFact:
    person_call: int
    native_period: int
    person_ticket: int
    airport_counter: bool


@dataclass(frozen=True)
class NativePresentationRow:
    analyst_root: int
    analyst_original_key: tuple
    fact: NativePresentationFact


@dataclass(frozen=True)
class ProofReentryView:
    native_period: int
    compiled_opportunity_utc: str
    earlier_presentation_calls: tuple
    native_proof_on_file: bool
    native_counter_this_period: bool
    scope: str = "native-presentation-call-history-before-current-reentry-marks-only"


@dataclass(frozen=True)
class ProofReentryRow:
    analyst_root: int
    analyst_original_key: tuple
    view: ProofReentryView


@dataclass(frozen=True)
class ProofInformationPeriod:
    period: int
    reentry_rows: tuple


@dataclass(frozen=True)
class ProofInformationSnapshot:
    native: object
    calls: tuple
    periods: tuple
    pending: object
    presentation_capture_failed: bool
    spec_hash: str
    rule: str = RULE


class ProofInformationMarketEngine(OwnHistoryInformationMarketEngine):
    """One inherited lifecycle; each successful native presentation kept once.

    Duplicate successful calls are distinct actions even when the native proof
    set deduplicates them. No new passenger, proof eligibility or choice rule.
    """

    def __init__(self, *args, **kwargs):
        self._proof_calls = []
        self._proof_periods = []
        self._pending_proof = None
        self._presentation_capture_failed = False
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._proof_calls = []
        self._proof_periods = []
        self._pending_proof = None
        self._presentation_capture_failed = False
        return super().reset(seed, tape=tape)

    def _require_complete_proof_capture(self):
        if self._pending_proof is not None or self._presentation_capture_failed:
            raise ValueError(
                "incomplete presentation capture; no query/continuation/retry"
            )

    def _capture_presentation(self, booking, proof):
        root = booking.record_id
        if root not in self._own_keys or self._own_tickets[root][0] != root:
            raise ValueError("prospective original ticket required; no reconstruction")
        ordinal = sum(row.analyst_root == root for row in self._proof_calls)
        self._proof_calls.append(
            NativePresentationRow(
                root,
                self._own_keys[root],
                NativePresentationFact(ordinal, proof.period, 0, proof.airport_counter),
            )
        )

    def present_cancelled_itinerary(self, booking, *, airport_counter=False):
        self._require_complete_proof_capture()
        proof = super().present_cancelled_itinerary(
            booking, airport_counter=airport_counter
        )
        # Parent success can already have mutated native proof state. A later
        # capture failure must not publish a positive partial journal or retry.
        self._presentation_capture_failed = True
        self._capture_presentation(booking, proof)
        self._presentation_capture_failed = False
        return proof

    def _marks(self, passenger, period):
        if self._pending_proof is None or self._presentation_capture_failed:
            raise ValueError("genuine complete prospective proof context required")
        root = passenger.original.record_id
        facts = tuple(r.fact for r in self._proof_calls if r.analyst_root == root)
        proved = root in self._presented
        counter = (root, period) in self._counter_visits
        if proved != bool(facts) or counter != any(
            f.native_period == period and f.airport_counter for f in facts
        ):
            raise RuntimeError("native proof status and prospective calls differ")
        self._pending_proof.append(
            ProofReentryRow(
                root,
                self._own_keys[root],
                ProofReentryView(
                    period,
                    self.demand_scenario.clock.at(period).isoformat(),
                    facts,
                    proved,
                    counter,
                ),
            )
        )
        return super()._marks(passenger, period)

    def step(self, actions=None, *, _requests=None, _choices=None):
        self._require_complete_proof_capture()
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        if self.ordinary_priority is not None or self._pending_own is not None:
            raise ValueError("complete chronological own-information path required")
        for name in ("booking_clock", "shock_clock"):
            clock = getattr(self, name, None)
            if clock is not None and clock != self.demand_scenario.clock:
                raise ValueError("native and compiled proof-information clocks differ")
        period = self.world.t
        self._pending_proof = []
        result = super().step(actions)
        own = self.own_information_periods()[-1]
        if tuple(
            (r.analyst_root, r.view.native_period) for r in own.reentry_rows
        ) != tuple((r.analyst_root, r.view.native_period) for r in self._pending_proof):
            raise RuntimeError("complete native own and proof attempts differ")
        self._proof_periods.append(
            ProofInformationPeriod(period, tuple(self._pending_proof))
        )
        self._pending_proof = None
        return result

    def presentation_calls(self):
        self._require_complete_proof_capture()
        return tuple(self._proof_calls)

    def proof_information_periods(self):
        self._require_complete_proof_capture()
        return tuple(self._proof_periods)

    def snapshot(self):
        return ProofInformationSnapshot(
            super().snapshot(),
            tuple(self._proof_calls),
            tuple(self._proof_periods),
            None if self._pending_proof is None else tuple(self._pending_proof),
            self._presentation_capture_failed,
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not ProofInformationSnapshot:
            raise TypeError("combined proof-information snapshot required")
        if snapshot.rule != RULE or snapshot.spec_hash != self.compiled.spec_hash:
            raise ValueError("proof-information capture contract differs")
        super().restore(snapshot.native)
        self._proof_calls = list(snapshot.calls)
        self._proof_periods = list(snapshot.periods)
        self._pending_proof = (
            None if snapshot.pending is None else list(snapshot.pending)
        )
        self._presentation_capture_failed = snapshot.presentation_capture_failed


def proof_information_view(row):
    """Owned narrow native-action projection, no analyst IDs or private marks."""
    if type(row) is not ProofReentryRow:
        raise TypeError("prospective proof re-entry row required")
    return _encode(jsonable(row.view))


def proof_information_key(row):
    payload = json.dumps(
        proof_information_view(row),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def proof_information_record(engine):
    """Read NEW complete receipts only; no native or upstream producer replay."""
    if type(engine) is not ProofInformationMarketEngine:
        raise TypeError("prospective proof-information producer required")
    periods = engine.proof_information_periods()
    return {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "capture_rule": RULE,
        "analyst_context": _encode(
            jsonable(
                {
                    "native_spec": engine.compiled.spec,
                    "spec_hash": engine.compiled.spec_hash,
                    "presentation_calls": engine.presentation_calls(),
                    "reentry_links": tuple(
                        tuple(
                            (r.analyst_root, r.analyst_original_key)
                            for r in p.reentry_rows
                        )
                        for p in periods
                    ),
                }
            )
        ),
        "ordinary_views_by_period": tuple(
            tuple(ordinary_information_view(r) for r in p.rows)
            for p in engine.information_periods()
        ),
        "own_views_by_period": tuple(
            tuple(own_reentry_information_view(r) for r in p.reentry_rows)
            for p in engine.own_information_periods()
        ),
        "proof_views_by_period": tuple(
            {
                "period": p.period,
                "views": tuple(proof_information_view(r) for r in p.reentry_rows),
                "view_keys": tuple(proof_information_key(r) for r in p.reentry_rows),
            }
            for p in periods
        ),
        "native_presentation_calls_captured": True,
        "actual_received_disclosure_complete_knowledge_or_legal_PNR_authenticated": False,
        "current_stage_menu_private_marks_probability_or_lifecycle_CS_exported": False,
        "native_actor_choice_observations_or_default_changed": False,
        "scientific_budget": None,
    }
