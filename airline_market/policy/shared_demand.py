"""A.5/B0/B1/V9: prospective shared Poisson potential envelope for demand contrasts.

One finite declared demand family on an identical non-demand native spec. Native
ExogenousTape.generate draws the envelope once. Projection changes acceptance
scores only, retains every potential row/slot and shares frozen auxiliary arrays.
No old tapes/results are edited; no kernel/planner or empirical adoption implied.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from airline_market.core.hashing import spec_hash
from airline_market.core.rng import StreamRegistry
from airline_market.sim.exogenous import demand_multiplier
from airline_market.sim.tape import ExogenousTape, segment_weights
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule


def _caps(compiled):
    s = compiled.spec
    if s.demand.mode != "poisson":
        raise ValueError("explicit native Poisson family required")
    multipliers = tuple(
        demand_multiplier(compiled, i) for i in range(len(s.exogenous.states))
    )
    if any(not math.isfinite(x) or x < 0 for x in multipliers) or max(multipliers) <= 0:
        raise ValueError(
            "nonnegative native multipliers with positive maximum required"
        )
    maximum = max(multipliers)
    values = np.asarray(
        [
            [
                [
                    s.demand.intensity[m.id].value_at(t) * maximum * w
                    for w in segment_weights(compiled, t, j)
                ]
                for j, m in enumerate(s.markets)
            ]
            for t in range(s.horizon.n_periods)
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("finite nonnegative stratum rates required")
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class SharedPoissonBank:
    tape: object
    envelope_compiled: object
    members: tuple
    member_caps: tuple
    envelope_rates: object
    namespace: str

    def project(self, name):
        members = dict(self.members)
        if name not in members:
            raise KeyError("scenario absent from declared shared family")
        cap = dict(self.member_caps)[name]
        projected = []
        for t, marks in enumerate(self.tape.arrival_marks):
            rows = []
            for mark in marks:
                rate = cap[t, mark.market, mark.segment]
                envelope = self.envelope_rates[t, mark.market, mark.segment]
                if envelope <= 0:
                    raise RuntimeError("zero envelope generated a potential request")
                ratio = rate / envelope
                # A score of one always fails the native strict thinning test.
                # Scores are transformed acceptance coordinates, not new uniforms.
                score = 1.0 if ratio == 0 else min(1.0, mark.thinning / ratio)
                rows.append(replace(mark, thinning=score))
            projected.append(tuple(rows))
        return replace(self.tape, arrival_marks=tuple(projected))

    def record(self):
        return {
            "plan_id": "A.5/B0/B1/V9",
            "seed": self.tape.seed,
            "namespace": self.namespace,
            "envelope_spec_sha256": self.envelope_compiled.spec_hash,
            "member_spec_sha256": {k: c.spec_hash for k, c in self.members},
            "envelope_rates": self.envelope_rates.tolist(),
            "member_max_state_rates": {k: x.tolist() for k, x in self.member_caps},
            "potential_rows_per_period": tuple(len(x) for x in self.tape.arrival_marks),
            "coupling_scope": "same declared family, seed, physical/segment/class/quote/event schema; all potential rows retained",
            "projection_rng_calls": 0,
            "generation_raw_rng_calls": None,
            "arrival_counts_scope": "potential envelope counts, not accepted member requests or empirical exposure",
            "primary_B1_or_empirical_adoption": False,
            "old_tapes_rewritten": False,
        }


def shared_poisson_bank(members, *, seed, namespace):
    """Draw one dominating native envelope; project declared members without RNG.

    Members may change intensity schedules, segment weights, holiday/exogenous
    demand multipliers. All non-demand spec fields must agree, so capacity/event,
    choice/class, quote, background and transition draw coordinates stay fixed.
    Enlarging/changing this family creates a NEW bank; old bank identity is not
    promised across regenerated potential envelopes or different namespaces.
    """
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("explicit shared-envelope namespace required")
    raw = tuple(members.items()) if isinstance(members, Mapping) else tuple(members)
    if len({k for k, c in raw}) != len(raw):
        raise ValueError("distinct declared scenario names required")
    items = tuple(sorted(raw))
    if not items or any(not isinstance(k, str) or not k for k, c in items):
        raise ValueError("nonempty named compiled demand family required")
    base = items[0][1]
    non_demand = spec_hash(replace(base.spec, demand=None))
    if any(spec_hash(replace(c.spec, demand=None)) != non_demand for k, c in items):
        raise ValueError("all non-demand native spec fields must agree")
    caps = tuple((name, _caps(c)) for name, c in items)
    wanted = np.maximum.reduce([x for k, x in caps])
    T, _M, G = wanted.shape
    intensities = {}
    weights = {}
    for m, market in enumerate(base.spec.markets):
        totals = []
        mixtures = []
        for t in range(T):
            rates = wanted[t, m]
            total = float(rates.sum())
            mix = tuple(float(x / total) for x in rates) if total else (1.0 / G,) * G
            # Upward rounding ensures computed native envelope rates dominate
            # every member even when sum/divide/multiply round differently.
            positive = [float(x / w) for x, w in zip(rates, mix) if w]
            total = math.nextafter(max([total, *positive]), math.inf) if total else 0.0
            totals.append(total)
            mixtures.append(mix)
        intensities[market.id] = PeriodSchedule(tuple(range(T)), tuple(totals))
        weights[market.id] = PeriodSchedule(tuple(range(T)), tuple(mixtures))
    d = replace(
        base.spec.demand,
        intensity=intensities,
        segment_weights=weights,
        holiday_multiplier=1.0,
        exogenous_multipliers={e.name: 1.0 for e in base.spec.exogenous.states},
    )
    compiled = compile_market(replace(base.spec, demand=d))
    rates = _caps(compiled)
    if (rates < wanted).any():
        raise ValueError("floating native envelope does not dominate declared rates")
    tape = ExogenousTape.generate(
        compiled, seed, registry=StreamRegistry(seed, namespace)
    )
    return SharedPoissonBank(tape, compiled, items, caps, rates, namespace)
