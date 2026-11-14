"""A.8/B0/V9: native price-proposal/clipping evidence, no lottery admission.

The cooperative capture hook lies after native action pricing and before static
caps. The outer hook records subsequent static/trigger clipping from the same
call. Controllers/rules are never repeated to reconstruct hypothetical proposals.
Price-constraint binding does not establish excess demand or choose a cap group.
"""

import copy
from dataclasses import dataclass

from airline_market.policy.rationing_native import assert_pool_current
from airline_market.policy.triggered_dated import TriggeredDatedMarketEngine
from airline_market.sim.engine import MarketEngine


class _NativeProposalStage(MarketEngine):
    def _action_prices(self, i, action, obs):
        super()._action_prices(i, action, obs)
        self._raw_proposal = (
            obs.t,
            i,
            tuple(float(x) for x in self.world.carriers[i].posted_prices),
        )


@dataclass(frozen=True)
class ProductCapEvidence:
    product: str
    native_proposal: float
    static_ceiling: float | None
    after_static: float
    trigger_ceiling: float | None
    executed: float

    @property
    def static_clipped(self):
        return self.after_static < self.native_proposal

    @property
    def trigger_clipped(self):
        return self.executed < self.after_static

    @property
    def price_constraint_binding(self):
        return self.executed < self.native_proposal


@dataclass(frozen=True)
class PostingCapEvidence:
    period: int
    revision: int
    products: tuple[ProductCapEvidence, ...]

    @property
    def any_price_constraint_binding(self):
        return any(row.price_constraint_binding for row in self.products)


@dataclass(frozen=True)
class CapEvidenceSnapshot:
    native: object
    action_rows: tuple
    completed_postings: tuple
    posting_complete: bool
    cap_policy: object


class CapEvidenceMarketEngine(TriggeredDatedMarketEngine, _NativeProposalStage):
    """Same native physical engine with analyst-only evidence at pricing hooks.

    Complete posting receipts require every carrier's native proposal, including
    leaders posted by begin_period/observe. Failed or incomplete posting has no new
    completed receipt. There is no transactional rollback promise after failure.
    Reposting is a new native call/revision and records its actual new proposal.
    """

    def __init__(self, *args, **kwargs):
        self._cap_action_rows = {}
        self._cap_completed = []
        self._raw_proposal = None
        self._cap_posting_complete = False
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._cap_action_rows = {}
        self._cap_completed = []
        self._raw_proposal = None
        self._cap_posting_complete = False
        return super().reset(seed, tape=tape)

    def _action_prices(self, i, action, obs):
        self._cap_posting_complete = False
        self._raw_proposal = None
        history_start = len(self.trigger.history) if self.trigger else 0
        super()._action_prices(i, action, obs)
        period, carrier, raw = self._raw_proposal
        assert (period, carrier) == (obs.t, i)
        static = self.cap_policy.limits(period)
        trigger = {}
        if self.trigger:
            for row in self.trigger.history[history_start:]:
                if row["active"] and row["cap"] is not None:
                    trigger[row["product"]] = float(row["cap"])
        evidence = []
        for k, j in enumerate(self.compiled.products_of[i]):
            product = self.compiled.spec.products[j].id
            after_static = min(raw[k], static.get(product, raw[k]))
            actual = float(self.world.carriers[i].posted_prices[k])
            expected = min(after_static, trigger.get(product, after_static))
            if actual != expected:
                raise RuntimeError("native cap-stage evidence differs from execution")
            evidence.append(
                ProductCapEvidence(
                    product,
                    raw[k],
                    static.get(product),
                    after_static,
                    trigger.get(product),
                    actual,
                )
            )
        self._cap_action_rows[(period, i)] = tuple(evidence)
        self._raw_proposal = None

    def post_actions(self, actions=None):
        self._cap_posting_complete = False
        prices = super().post_actions(actions)
        t = self.world.t
        rows = {}
        for i in range(len(self.compiled.spec.carriers)):
            if (t, i) not in self._cap_action_rows:
                raise RuntimeError("complete native proposal coverage required")
            rows.update((row.product, row) for row in self._cap_action_rows[(t, i)])
        ordered = tuple(rows[p.id] for p in self.compiled.spec.products)
        if tuple(row.executed for row in ordered) != tuple(float(x) for x in prices):
            raise RuntimeError("completed posting differs from proposal evidence")
        self._cap_completed.append(
            PostingCapEvidence(
                t,
                1 + sum(r.period == t for r in self._cap_completed),
                ordered,
            )
        )
        self._cap_posting_complete = True
        return prices

    def cap_postings(self):
        return tuple(self._cap_completed)

    def pool_cap_evidence(self, pool):
        """Join the current pool to this engine's last completed native posting.

        Does not set pool.binding_cap: the affected request/cap group and lottery
        timing remain an explicit future admission design. No excess-demand or
        causal scarcity/welfare claim follows from clipping an observed proposal.
        """
        assert_pool_current(self, pool)
        if (
            not self._cap_posting_complete
            or not self._cap_completed
            or self._cap_completed[-1].period != self.world.t
        ):
            raise ValueError("completed native posting evidence required")
        row = self._cap_completed[-1]
        if tuple(x.executed for x in row.products) != pool.posted_prices:
            raise ValueError("pool and native completed posting differ")
        return row

    def snapshot(self):
        return CapEvidenceSnapshot(
            super().snapshot(),
            tuple(sorted(self._cap_action_rows.items())),
            tuple(self._cap_completed),
            self._cap_posting_complete,
            self.cap_policy,
        )

    def restore(self, snapshot):
        if type(snapshot) is not CapEvidenceSnapshot:
            raise TypeError("cap-evidence combined snapshot required")
        if snapshot.cap_policy != self.cap_policy:
            raise ValueError("snapshot public static cap contract differs")
        super().restore(snapshot.native)
        self._cap_action_rows = dict(snapshot.action_rows)
        self._cap_completed = list(snapshot.completed_postings)
        self._cap_posting_complete = snapshot.posting_complete
        self._raw_proposal = None

    def branch(self):
        other = CapEvidenceMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            trigger_rule=self.trigger.rule if self.trigger else None,
            mechanical_benchmark=self.trigger.benchmark if self.trigger else None,
            booking_clock=self.booking_clock,
            leg_departures=self.leg_departures,
            dated_rebooking=self.dated_rebooking,
            dated_menu=self.dated_menu,
            dated_reserve=self.dated_reserve,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other


def posting_record(row):
    """Analyst JSON of actual native proposal/clip stages; never actor input."""
    if type(row) is not PostingCapEvidence:
        raise TypeError("native PostingCapEvidence required")
    return {
        "plan_id": "A.8/B0/V9",
        "period": row.period,
        "revision": row.revision,
        "products": tuple(
            {
                **vars(x),
                "static_clipped": x.static_clipped,
                "trigger_clipped": x.trigger_clipped,
                "price_constraint_binding": x.price_constraint_binding,
            }
            for x in row.products
        ),
        "any_price_constraint_binding": row.any_price_constraint_binding,
        "excess_demand_measured": False,
        "lottery_group_adopted": False,
        "native_rationing_adopted": False,
    }
