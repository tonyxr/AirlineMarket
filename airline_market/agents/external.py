"""Explicit actions supplied by a focal environment or multi-carrier game."""

from .base import StatelessController


class ExternalController(StatelessController):
    """Carrier whose actions are supplied explicitly by an environment (Eqs. 18–19)."""

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        raise RuntimeError("external controller requires an explicit action")
