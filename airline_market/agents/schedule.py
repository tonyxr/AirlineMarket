"""Time-indexed coefficient schedules (Eq. 18)."""

from ..core.types import CoefficientAction
from .base import StatelessController


class TimeScheduleController(StatelessController):
    """Departure-period schedule of pricing coefficients (Eqs. 18–19)."""

    def __init__(self, thetas_by_period):
        """
        Bind declared inputs and initialize TimeScheduleController bookkeeping (Eqs.
        18–19).
        """
        self.thetas = tuple(tuple(x) for x in thetas_by_period)

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        return CoefficientAction(self.thetas[min(obs.t, len(self.thetas) - 1)])
