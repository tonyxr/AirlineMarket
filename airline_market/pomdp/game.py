"""Multi-carrier action interface; simultaneous base and staged leaders."""

from ..agents.external import ExternalController
from ..sim.engine import MarketEngine
from .objective import ObjectiveEvaluator


class MarketGame:
    """Multi-carrier cohort interface with explicit leader commitments (Eqs. 24–27)."""

    def __init__(self, compiled, arms=None):
        """Bind declared inputs and initialize MarketGame bookkeeping (Eqs. 24–27)."""
        from ..spec.model import InformationArmSpec

        self.compiled = compiled
        self.engine = MarketEngine(
            compiled,
            {i: ExternalController() for i in range(len(compiled.spec.carriers))},
        )
        self.arms = arms or {
            c.code: next(
                (a for a in compiled.spec.information if a.observer == c.code),
                InformationArmSpec(name="own", observer=c.code, requests=()),
            )
            for c in compiled.spec.carriers
        }
        self.objectives = {
            c.code: ObjectiveEvaluator(
                next(o for o in compiled.spec.objectives if o.carrier == c.code),
                compiled,
            )
            for c in compiled.spec.carriers
        }

    def observations(self):
        """
        Return only the carriers entitled to act at the current timing stage (Eqs.
        24–27).
        """
        s = self.compiled.spec
        visible = s.carriers
        if (
            s.timing.mode == "leader_follower"
            and not self.engine.terminal()
            and len(self.engine.world.posted_leaders) < len(s.timing.leaders)
        ):
            visible = tuple(c for c in s.carriers if c.code in s.timing.leaders)
        return {c.code: self.engine.observe(c.code, self.arms[c.code]) for c in visible}

    def reset(self, seed):
        """Start a new cohort and clear per-episode memory (Eqs. 24–27)."""
        self.pending = {}
        self.engine.reset(seed)
        return self.observations()

    def post_leaders(self, actions):
        """
        Commit leader actions before exposing their offers to followers (Eqs. 24–27).
        """
        if self.pending:
            raise ValueError("leaders have already committed")
        if set(actions) != set(self.compiled.spec.timing.leaders):
            raise ValueError("supply exactly the leader actions")
        self.pending = dict(actions)
        self.engine.begin_period(actions)
        return self.observations()

    def step(self, actions):
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        24–27).
        """
        if set(actions) & set(self.pending):
            raise ValueError("committed leader actions cannot be replaced")
        before = self.engine.world.clone()
        result = self.engine.step({**self.pending, **actions})
        self.pending = {}
        done = self.engine.terminal()
        rewards = {}
        for i, c in enumerate(self.compiled.spec.carriers):
            objective = self.objectives[c.code]
            r = objective.period_reward(i, before, self.engine.world)
            rewards[c.code] = r + (
                objective.terminal_reward(self.engine.world) if done else 0.0
            )
        return self.observations(), rewards, done, {"period": result.period}
