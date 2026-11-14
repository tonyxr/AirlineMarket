"""Poisson market-local taste storage with unchanged non-taste named streams.

The potential request market is exogenous and fixed before any policy acts.
Only that market's alternatives can enter its random-utility maximization.
Unused global alternatives therefore need no stored tastes. Taste draws stay
iid Gumbel, with a separately declared layout; dense-tape bit identity is not
claimed. All other tape fields retain their original stream and draw order.
"""

from dataclasses import dataclass
import numpy as np

from ..core.rng import StreamRegistry
from .exogenous import categorical, demand_multiplier
from .tape import ExogenousTape, ArrivalMark, segment_weights


@dataclass(frozen=True)
class MarketLocalGumbels:
    """Read-only ragged tastes indexed by potential request and declared market."""

    shape: tuple
    markets: tuple
    columns: tuple
    offsets: tuple
    values: tuple

    @property
    def nbytes(self):
        """Count stored tastes and integer index arrays, excluding Python overhead."""
        return sum(x.nbytes for x in (*self.markets, *self.offsets, *self.values))

    def __getitem__(self, key):
        """Expose exactly the engine's declared alternative vector, without mutation."""
        t, row, columns = key
        market = int(self.markets[t][row])
        if tuple(columns) != self.columns[market]:
            raise ValueError("market-local tape access must use the request's declared choice set")
        a, b = self.offsets[t][row:row + 2]
        return self.values[t][a:b]


def generate_market_local(compiled, seed, registry=None):
    """Generate an immutable Poisson tape with no global unused taste allocation."""
    reg = registry or StreamRegistry(seed, "exo")
    s, T = compiled.spec, compiled.spec.horizon.n_periods
    if s.demand.mode != "poisson":
        raise ValueError("market-local revision is defined for marked Poisson requests")
    counts = np.zeros((T, len(s.markets), len(s.segments)), dtype=np.int64)
    marks = []
    max_multiplier = max(demand_multiplier(compiled, e) for e in range(len(s.exogenous.states)))
    for t in range(T):
        period = []
        for m, market in enumerate(s.markets):
            for g, weight in enumerate(segment_weights(compiled, t, m)):
                rng = reg.generator(20, t, m, g)
                n = int(rng.poisson(s.demand.intensity[market.id].value_at(t)
                                    * max_multiplier * weight))
                counts[t, m, g] = n
                for vals in rng.random((n, 3)):
                    period.append(ArrivalMark(float(vals[0]), m, g,
                                              float(vals[1]), float(vals[2])))
        period.sort(key=lambda a: (a.time, a.market, a.segment))
        marks.append(tuple(period))
    max_requests = max(1, max(map(len, marks)))
    columns = tuple((*ps, len(s.products)) for ps in compiled.market_products)
    market_rows, offsets, values = [], [], []
    rng = reg.generator(3)
    for period in marks:
        ids = np.array([mark.market for mark in period], dtype=np.int64)
        boundaries = np.concatenate(([0], np.cumsum([len(columns[m]) for m in ids], dtype=np.int64)))
        draws = rng.gumbel(size=int(boundaries[-1]))
        for array in (ids, boundaries, draws):
            array.setflags(write=False)
        market_rows.append(ids)
        offsets.append(boundaries)
        values.append(draws)
    local = MarketLocalGumbels((T, max_requests, len(s.products) + 1),
                              tuple(market_rows), columns, tuple(offsets), tuple(values))
    bg_rng = reg.generator(5)
    background = bg_rng.random((T, len(s.background)))
    initial = np.array([b.capacity.support[categorical(float(u), b.capacity.probabilities)]
                       for b, u in zip(s.background, bg_rng.random(len(s.background)))], dtype=np.int64)
    targets = max([len(q.target_products) for q in s.quotes.requests] or [1])
    tape = ExogenousTape(seed, reg.generator(6).random(T),
        reg.generator(7).random(len(s.legs)), reg.generator(1).random((T, 3)),
        reg.generator(2).random((T, max_requests)), counts, tuple(marks), local,
        reg.generator(4).normal(size=(T, max_requests, len(s.carriers) + 1)),
        reg.generator(8).random((T, len(s.capacity_process.events))), background, initial,
        reg.generator(9).random((T + 1, len(s.quotes.requests), targets, 2)),
        float(reg.generator(10).random()))
    tape.freeze()
    return tape
