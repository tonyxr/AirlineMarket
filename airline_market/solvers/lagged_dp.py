"""Opt-in exact lag-one solver; existing campaign backends stay unchanged."""

from ..sim.lagged_finite import LaggedFinitePricingModel
from .conditional_tail import ConditionalTailSolver
from .exact_dp import ExactBeliefDP
from .finite_dp import FiniteBeliefSolver
from .oracle_dp import OracleDP


def history_dp(compiled, focal, arm, *, max_nodes=2_000_000,
               rival_controllers=None, conditional_tail=True):
    """Keep general initial observations and use certified augmented-state recursion."""
    kwargs = {"backend": "reference", "max_nodes": max_nodes,
              "rival_controllers": rival_controllers}
    dp = (OracleDP(compiled, focal, **kwargs) if arm == "full_state"
          else ExactBeliefDP(compiled, focal, arm, **kwargs))
    model = LaggedFinitePricingModel(dp.compiled, focal, dp.arm, rival_controllers)
    solver = ConditionalTailSolver if conditional_tail else FiniteBeliefSolver
    dp.finite = solver(model, dp.grid, max_nodes)
    dp.backend = "compact_history_lag1"
    dp.backend_reason = None
    return dp
