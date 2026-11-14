"""Fixed coefficients retain dynamic state features (Eq. 18)."""

from ..core.types import CoefficientAction
from .base import StatelessController


class FixedCoefficientController(StatelessController):
    """
    Fixed coefficients applied to dynamic observable pricing features (Eqs. 18–19).
    """

    def __init__(self, theta):
        """
        Bind declared inputs and initialize FixedCoefficientController bookkeeping (Eqs.
        18–19).
        """
        self.theta = tuple(map(float, theta))

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        return CoefficientAction(self.theta)
