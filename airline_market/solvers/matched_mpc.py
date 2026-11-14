"""Scenario-indexed planning CRN for the F5 selection comparison.

The reference planner draws all hypotheses before all forecast seeds, so
changing its sample count moves the forecast stream. This study adapter
indexes each complete hypothesis/forecast pair independently. All, split,
protection and racing therefore use identical selection prefixes; reporting
uses a separate family. This changes the planning RNG coupling, not the
physical model, posterior or incumbent-switching rule.
"""

from ..core.rng import StreamRegistry
from ..sim.tape import ExogenousTape
from .mpc import ScenarioMPC


class MatchedScenarioMPC(ScenarioMPC):
    """Use the same indexed planning pairs across budgets and racing rounds."""

    def __init__(self, *args, **kwargs):
        """Retain the reference planner and expose only draw-index diagnostics."""
        super().__init__(*args, **kwargs)
        self.draw_requests = []

    def _scenarios(self, obs, episode_index, count, stage):
        """Construct independent planning pairs without count-dependent offsets."""
        if stage == 1:
            family, offset = 1, 0
        else:
            family = 0
            offset = (
                0
                if stage <= 2
                else min(
                    self.config.racing_initial_scenarios * 2 ** (stage - 3),
                    self.config.n_scenarios,
                )
            )
        period = 0 if self.config.reuse_scenarios_across_periods else obs.t
        result = []
        for index in range(offset, offset + count):
            # Namespace is explicit and distinct from physical evaluation tapes.
            rng = StreamRegistry(
                self.streams.root_seed, "plan/F5/indexed-pairs-v1"
            ).generator(episode_index, period, family, index)
            hidden = self.belief.sample(1, rng)[0]
            engine = self.kernel.engine(hidden.kernel, self.focal)
            engine.tape = ExogenousTape.generate(
                self.compiled, int(rng.integers(2**63))
            )
            result.append(engine)
        self.draw_requests.append(
            {
                "period": obs.t,
                "episode": episode_index,
                "family": family,
                "offset": offset,
                "count": count,
            }
        )
        return result
