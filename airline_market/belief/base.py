"""Finite hidden hypotheses; all states are model-generated, never truth reads."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, Any
import numpy as np
from ..sim.kernel import KernelCarrierState, KernelState


@dataclass(frozen=True)
class HiddenState:
    """
    Immutable rival inventory/count, exogenous and nuisance-state hypothesis (Eqs.
    24–25, 30).
    """

    rivals: tuple[KernelCarrierState, ...]
    background: tuple[int, ...]
    exo_latent: int | None
    history: tuple = ()
    kernel: KernelState | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    @cached_property
    def _cached_hash(self):
        """Cache the immutable hidden hypothesis's structural hash (Eqs. 24–25, 30)."""
        return hash((self.rivals, self.background, self.exo_latent, self.history))

    def __hash__(self):
        """Return the structural hash of this immutable value (Eqs. 24–25, 30)."""
        return self._cached_hash

    def sort_key(self):
        """Return a stable ordering key for posterior sampling (Eqs. 24–25, 30)."""
        return (
            tuple(
                (r.capacity, r.counts, r.previous_prices, repr(r.controller_memory))
                for r in self.rivals
            ),
            self.background,
            -1 if self.exo_latent is None else self.exo_latent,
            repr(self.history),
        )

    def __lt__(self, other):
        """Compare stable hypothesis ordering keys (Eqs. 24–25, 30)."""
        return self.sort_key() < other.sort_key()


class Belief(Protocol):
    """
    Finite or particle posterior protocol with observation-only updates (Eqs. 24–25,
    30).
    """

    def reset(self, first_obs):
        """
        Start a new posterior from the first permitted observation (Eqs. 24–25, 30).
        """
        ...

    def update(self, before, action, after):
        """
        Condition the maintained posterior on the observed action/outcome sequence (Eqs.
        24–25, 30).
        """
        ...

    def sample(self, n: int, rng: np.random.Generator) -> list[HiddenState]:
        """
        Draw model hypotheses from the maintained posterior using the supplied RNG (Eqs.
        24–25, 30).
        """
        ...

    def leg_marginals(self):
        """
        Return each rival leg's probability vector over residual seats (Eqs. 24–25, 30).
        """
        ...

    def count_marginals(self):
        """
        Return each rival product's posterior active-booking count distribution (Eqs.
        24–25, 30).
        """
        ...

    def event_probability(self, event):
        """
        Integrate a declared joint event over the maintained posterior (Eqs. 24–25, 30).
        """
        ...

    def entropy(self) -> float:
        """
        Return Shannon entropy of the represented hidden-state distribution (Eqs. 24–25,
        30).
        """
        ...

    def key(self):
        """
        Return the deterministic memoization key for the represented state (Eqs. 24–25,
        30).
        """
        ...

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 24–25, 30).
        """
        ...

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 24–25, 30)."""
        ...
