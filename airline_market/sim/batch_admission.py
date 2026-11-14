"""Two-stage fixed-fare admission clock with unchanged native booking ledgers."""

from dataclasses import dataclass

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import Request
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine
from airline_market.sim.ledger import book


@dataclass(frozen=True)
class DemandTape:
    """One immutable independently drawn class-demand pair, shared across controls."""

    seed: int
    low: int
    high: int


def draw_tape(seed, low_n, low_p, high_n, high_p):
    """Separate named class streams; no controller receives the future count."""
    return DemandTape(
        seed,
        int(
            StreamRegistry(seed, "littlewood_low_demand")
            .generator()
            .binomial(low_n, low_p)
        ),
        int(
            StreamRegistry(seed, "littlewood_high_demand")
            .generator()
            .binomial(high_n, high_p)
        ),
    )


class BatchAdmissionEngine(FixedFareAdmissionEngine):
    """At t=0 reveal low requests individually, at t=1 reveal high requests."""

    def __init__(self, compiled):
        if compiled.spec.horizon.n_periods != 2 or len(compiled.spec.products) != 2:
            raise ValueError("two stages and two fare classes required")
        super().__init__(compiled, [[0.0, 0.0], [0.0, 0.0]])
        self.rank = 0

    def reset(self):
        """Reuse the verified native empty-world construction, with request rank0."""
        super().reset()
        self.rank = 0

    def take_request(self, accept):
        """Apply one current-stage decision; rejection makes no cash/stock entry."""
        j = self.world.t  # source product order: LOW, HIGH
        obs = self.observe(j)
        if type(accept) is not bool:
            raise ValueError("boolean admission required")
        booking = None
        if accept:
            request = Request(
                self.rank, int(self.compiled.product_market[j]), 0, 0, 0, 0
            )
            booking = book(self.world, 0, j, obs.fixed_fare_USD, request, self.compiled)
        self.world.request_log.append(
            {"period": j, "rank": self.rank, "requested_product": j, "admitted": accept}
        )
        self.rank += 1
        return booking

    def clone(self):
        """Clone the native ledger and the within-stage request clock."""
        other = super().clone()
        other.rank = self.rank
        return other

    def next_stage(self):
        """Advance only after all requests of the current class were revealed."""
        if self.world.t >= 2:
            raise ValueError("episode is terminal")
        self.world.t += 1
        self.rank = 0
        self.world.phase = (
            PeriodPhase.TERMINAL if self.world.t == 2 else PeriodPhase.PRE_DECISION
        )


def protection_action(obs, protected_seats):
    """Use only public current stock/class; no demand tape or future count input."""
    return bool(
        obs.can_accept
        and obs.own_inventory[0]
        > (protected_seats if obs.requested_product == 0 else 0)
    )


def run_tape(engine, tape, protected_seats):
    """Execute every realized request against native ledger admission."""
    engine.reset()
    for count in (tape.low, tape.high):
        for _ in range(count):
            obs = engine.observe(engine.world.t)
            engine.take_request(protection_action(obs, protected_seats))
        engine.next_stage()
    return engine.world
