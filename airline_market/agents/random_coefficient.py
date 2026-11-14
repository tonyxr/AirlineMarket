"""Named-stream random coefficient stress policy for simulation scaling (F3)."""

from copy import deepcopy
import numpy as np

from .base import StatelessController
from ..core.hashing import canonical_json
from ..core.types import CoefficientAction


class RandomCoefficientController(StatelessController):
    """Draw bounded coefficients from the controller's own RNG (Eq. 18).

    The controller does not receive hidden state or an exogenous tape. Its
    random coins are restored with snapshots; it is sampling-only.
    """

    exact_compatible = False
    policy_class = "random_coefficient_stress"

    def reset(self, ctx, rng) -> None:
        """Bind immutable bounds and the engine's named carrier-control stream."""
        super().reset(ctx, rng)
        self.bounds = np.asarray(ctx.pricing.theta_bounds, dtype=float)

    def act(self, obs) -> CoefficientAction:
        """Post random coefficients without reading another carrier's state."""
        return CoefficientAction(
            tuple(self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1]))
        )

    def memory(self):
        """Expose only the controller's own finite RNG state to simulator memory."""
        return canonical_json(self.rng.bit_generator.state)

    def snapshot(self):
        """Preserve random-control coins for exact replay on the same tape."""
        return deepcopy(self.rng.bit_generator.state)

    def restore(self, snapshot) -> None:
        """Restore the controller's own coins without mutating shared model data."""
        self.rng.bit_generator.state = deepcopy(snapshot)
