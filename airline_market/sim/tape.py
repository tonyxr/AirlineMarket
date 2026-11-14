"""Read-only policy-independent random tapes (Eqs. 4–6)."""

from dataclasses import dataclass
import numpy as np
from ..core.rng import StreamRegistry
from .exogenous import categorical, demand_multiplier


@dataclass(frozen=True)
class ArrivalMark:
    """
    Policy-independent potential Poisson request and thinning/taste draws (Eqs. 3–16).
    """

    time: float
    market: int
    segment: int
    thinning: float
    class_uniform: float


def segment_weights(compiled, t: int, market: int):
    """Resolve the period/market-specific passenger segment mixture (Eqs. 3–16)."""
    s = compiled.spec.demand.segment_weights
    return (
        s.value_at(t)
        if hasattr(s, "value_at")
        else s[compiled.spec.markets[market].id].value_at(t)
    )


@dataclass(frozen=True)
class ExogenousTape:
    """
    Read-only pre-indexed demand, taste, shock and shopping randomness (Eqs. 3–16).
    """

    seed: int
    exo_uniform: np.ndarray
    initial_capacity_uniform: np.ndarray
    request_uniforms: np.ndarray
    class_uniform: np.ndarray
    arrival_counts: np.ndarray | None
    arrival_marks: tuple
    gumbel: np.ndarray
    mixed_draws: np.ndarray
    capacity_uniforms: np.ndarray
    background_uniforms: np.ndarray
    background_initial: np.ndarray
    quote_uniforms: np.ndarray
    initial_exo_uniform: float = 0.0

    @classmethod
    def generate(cls, compiled, seed: int, registry: StreamRegistry | None = None):
        """
        Pre-index every exogenous draw in independent named random streams (Eqs. 3–16).
        """
        reg = registry or StreamRegistry(seed, "exo")
        s = compiled.spec
        T = s.horizon.n_periods
        counts = None
        marks = []
        max_requests = 1
        if s.demand.mode == "poisson":
            M, G = len(s.markets), len(s.segments)
            counts = np.zeros((T, M, G), dtype=np.int64)
            max_multiplier = max(
                demand_multiplier(compiled, e) for e in range(len(s.exogenous.states))
            )
            for t in range(T):
                period = []
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
                        for vals in rng.random((n, 3)):
                            period.append(
                                ArrivalMark(
                                    float(vals[0]), m, g, float(vals[1]), float(vals[2])
                                )
                            )
                period.sort(key=lambda a: (a.time, a.market, a.segment))
                marks.append(tuple(period))
            max_requests = max(1, max(map(len, marks)))
        else:
            marks = [() for _ in range(T)]
        request = reg.generator(1).random((T, 3))
        classes = reg.generator(2).random((T, max_requests))
        gumbel = reg.generator(3).gumbel(size=(T, max_requests, len(s.products) + 1))
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
            tuple(marks),
            gumbel,
            mixed,
            reg.generator(8).random((T, len(s.capacity_process.events))),
            bg_uniform,
            bg_initial,
            reg.generator(9).random((T + 1, len(s.quotes.requests), n_targets, 2)),
            float(reg.generator(10).random()),
        )
        result.freeze()
        return result

    def freeze(self):
        """
        Make all tape arrays read-only for replay and counterfactual branches (Eqs.
        3–16).
        """
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)


class LegacyToyTapeAdapter:
    """
    Bit-order-preserving adapter for the frozen prototype's random tape (Eqs. 3–16).
    """

    @staticmethod
    def from_legacy(legacy_noise_tape, compiled=None, shock_step: int | None = None):
        """
        Adapt the frozen prototype draws without changing their ordering (Eqs. 3–16).
        """
        if compiled is None:
            from ..spec.library_builders import toy_hub
            from ..spec.compiled import compile_market

            compiled = compile_market(toy_hub())
        T = compiled.spec.horizon.n_periods
        step = shock_step or compiled.spec.metadata["legacy_config"]["shock_step"]
        exo = np.full(T, 0.5)
        exo[step - 1] = 1.0 - legacy_noise_tape.weather_uniform
        capacity = np.zeros(len(compiled.spec.legs))
        capacity[compiled.leg_index["B_HD"]] = legacy_noise_tape.initial_uniform
        targets = max(len(q.target_products) for q in compiled.spec.quotes.requests)
        result = ExogenousTape(
            0,
            exo,
            capacity,
            legacy_noise_tape.requests.copy(),
            np.zeros((T, 1)),
            None,
            tuple(() for _ in range(T)),
            legacy_noise_tape.gumbels[:, None, :].copy(),
            np.zeros((T, 1, len(compiled.spec.carriers) + 1)),
            np.zeros((T, len(compiled.spec.capacity_process.events))),
            legacy_noise_tape.reference[:, None].copy(),
            np.array([legacy_noise_tape.reference_initial]),
            np.full((T + 1, len(compiled.spec.quotes.requests), targets, 2), 0.5),
        )
        result.freeze()
        return result
