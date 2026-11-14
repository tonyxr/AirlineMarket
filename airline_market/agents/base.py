"""Controllers receive observations, not engine/tape references (Eq. 17)."""

from dataclasses import dataclass
from typing import Any, Hashable, Protocol
import copy
import numpy as np
from ..core.types import Action


@dataclass(frozen=True)
class ControllerContext:
    """
    Read-only model inputs, objective and permitted information for one controller (Eqs.
    18–19).
    """

    compiled: Any
    carrier: int
    pricing: Any
    objective: Any
    arm: Any = None


class Controller(Protocol):
    """Observation-only controller protocol with explicit finite memory (Eqs. 18–19)."""

    carrier: int
    policy_class: str
    exact_compatible: bool

    def reset(self, ctx: ControllerContext, rng: np.random.Generator) -> None:
        """Start a new cohort and clear per-episode memory (Eqs. 18–19)."""
        ...

    def act(self, obs) -> Action:
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        ...

    def observe_outcome(self, outcome) -> None:
        """
        Update controller bookkeeping from its own realized period outcome (Eqs. 18–19).
        """
        ...

    def memory(self) -> Hashable:
        """
        Return the controller's hashable finite memory, without world references (Eqs.
        18–19).
        """
        ...

    def snapshot(self) -> Any:
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 18–19).
        """
        ...

    def restore(self, s: Any) -> None:
        """Restore a previously captured execution/filter snapshot (Eqs. 18–19)."""
        ...


class StatelessController:
    """Controller base with no episode memory beyond declared context (Eqs. 18–19)."""

    policy_class = "coefficient"
    exact_compatible = True

    def reset(self, ctx, rng):
        """Start a new cohort and clear per-episode memory (Eqs. 18–19)."""
        self.ctx = ctx
        self.carrier = ctx.carrier
        self.rng = rng

    def observe_outcome(self, outcome):
        """
        Update controller bookkeeping from its own realized period outcome (Eqs. 18–19).
        """
        pass

    def memory(self):
        """
        Return the controller's hashable finite memory, without world references (Eqs.
        18–19).
        """
        return ()

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 18–19).
        """
        return ()

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 18–19)."""
        pass
