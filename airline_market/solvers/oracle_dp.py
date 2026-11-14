"""Full-current-state reference; future uncertainty remains stochastic."""

from dataclasses import replace
from .exact_dp import ExactBeliefDP
from ..spec import compile_market


class OracleDP(ExactBeliefDP):
    """
    Full-current-state reference with future random draws still unknown (Eqs. 26–32).
    """

    def __init__(self, compiled, focal, **kwargs):
        """Bind declared inputs and initialize OracleDP bookkeeping (Eqs. 26–32)."""
        from ..spec.model import InformationArmSpec

        requests = next(
            (
                a.requests
                for a in compiled.spec.information
                if a.observer == focal and not a.privileged
            ),
            (),
        )
        spec = replace(
            compiled.spec,
            information=compiled.spec.information
            + (
                InformationArmSpec(
                    name="full_state_dp",
                    observer=focal,
                    requests=requests,
                    privileged=True,
                ),
            ),
        )
        super().__init__(compile_market(spec), focal, "full_state_dp", **kwargs)
