"""Exact original tape values for a single simulated period, with bounded storage.

The horizon-wide Poisson count scan remains: its maximum controls original taste
array dimensions. Unused marks and future taste rows are never materialized.
"""

import numbers
from dataclasses import dataclass

import numpy as np

from airline_market.core.rng import StreamRegistry
from airline_market.sim.exogenous import categorical, demand_multiplier
from airline_market.sim.tape import ArrivalMark, ExogenousTape, segment_weights


@dataclass(frozen=True)
class PeriodRow:
    """A read-only, explicitly partial row; unavailable periods raise on access."""

    period: int
    value: object
    shape: tuple

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        first, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if not isinstance(first, numbers.Integral) or first != self.period:
            raise IndexError("one-step tape row is not available for this period")
        return self.value[rest] if rest else self.value


def _prefix_row(rng, period, shape):
    """Consume the identical flattened Gumbel sequence without holding the prefix."""
    row = None
    for _ in range(period + 1):
        row = rng.gumbel(size=shape)
    row.setflags(write=False)
    return row


class OneStepTape(ExogenousTape):
    """Opt-in native one-step proposals; not a full rollout, reset or MPC tape."""

    @classmethod
    def generate(cls, compiled, seed, period, registry=None):
        reg = registry or StreamRegistry(seed, "exo")
        s = compiled.spec
        T = s.horizon.n_periods
        if not 0 <= period < T:
            raise ValueError("one-step tape period must be inside the horizon")
        counts = None
        current_marks = []
        max_requests = 1
        if s.demand.mode == "poisson":
            counts = np.zeros((T, len(s.markets), len(s.segments)), dtype=np.int64)
            max_multiplier = max(
                demand_multiplier(compiled, e) for e in range(len(s.exogenous.states))
            )
            for t in range(T):
                total = 0
                for m, market in enumerate(s.markets):
                    for g, w in enumerate(segment_weights(compiled, t, m)):
                        intensity = (
                            s.demand.intensity[market.id].value_at(t)
                            * max_multiplier
                            * w
                        )
                        rng = reg.generator(20, t, m, g)
                        n = int(rng.poisson(intensity))
                        counts[t, m, g] = n
                        total += n
                        if t == period:
                            for vals in rng.random((n, 3)):
                                current_marks.append(
                                    ArrivalMark(
                                        float(vals[0]),
                                        m,
                                        g,
                                        float(vals[1]),
                                        float(vals[2]),
                                    )
                                )
                max_requests = max(max_requests, total)
            current_marks.sort(key=lambda a: (a.time, a.market, a.segment))
        request = reg.generator(1).random((T, 3))
        classes = reg.generator(2).random((T, max_requests))
        row = _prefix_row(reg.generator(3), period, (max_requests, len(s.products) + 1))
        mixed = reg.generator(4).normal(size=(T, max_requests, len(s.carriers) + 1))
        bg_rng = reg.generator(5)
        bg_uniform = bg_rng.random((T, len(s.background)))
        bg_initial = np.array(
            [
                b.capacity.support[categorical(float(u), b.capacity.probabilities)]
                for b, u in zip(s.background, bg_rng.random(len(s.background)))
            ],
            dtype=np.int64,
        )
        n_targets = max([len(q.target_products) for q in s.quotes.requests] or [1])
        result = cls(
            seed,
            reg.generator(6).random(T),
            reg.generator(7).random(len(s.legs)),
            request,
            classes,
            counts,
            PeriodRow(period, tuple(current_marks), (T,)),
            PeriodRow(period, row, (T, max_requests, len(s.products) + 1)),
            mixed,
            reg.generator(8).random((T, len(s.capacity_process.events))),
            bg_uniform,
            bg_initial,
            reg.generator(9).random((T + 1, len(s.quotes.requests), n_targets, 2)),
            float(reg.generator(10).random()),
        )
        result.freeze()
        return result
