"""One observable carrier optimizing against fixed rival mappings (Eq. 26)."""

import copy
import numpy as np
from ..core.types import CoefficientAction, PriceAction, HoldAction
from ..agents.external import ExternalController
from ..agents.price_rule import AdditivePriceRule
from ..sim.engine import MarketEngine
from .objective import ObjectiveEvaluator


class FocalCarrierEnv:
    """
    One observable carrier optimizing against explicitly supplied rival mappings (Eqs.
    24–27).
    """

    def __init__(
        self,
        compiled,
        focal: str,
        arm: str,
        rival_controllers=None,
        *,
        action_mode="grid",
        seed_plan=None,
        critic_privileged=False,
    ):
        """
        Bind declared inputs and initialize FocalCarrierEnv bookkeeping (Eqs. 24–27).
        """
        self.compiled = compiled
        self.focal = focal
        self.i = compiled.carrier_index[focal]
        self.arm = next(
            a
            for a in compiled.spec.information
            if a.name == arm and a.observer == focal
        )
        self.pricing = next(p for p in compiled.spec.pricing if p.carrier == focal)
        self.price_rule = AdditivePriceRule(compiled, self.pricing, self.arm)
        controllers = dict(rival_controllers or {})
        controllers[self.i] = ExternalController()
        self.engine = MarketEngine(compiled, controllers)
        self.action_mode = action_mode
        objective = next(o for o in compiled.spec.objectives if o.carrier == focal)
        self.objective = ObjectiveEvaluator(objective, compiled)
        self.critic_privileged = critic_privileged

    def reset(self, seed: int):
        """Start a new cohort and clear per-episode memory (Eqs. 24–27)."""
        self.engine.reset(seed)
        obs = self.engine.observe(self.i, self.arm)
        return obs, self._info()

    def _info(self):
        """
        Return outcome diagnostics and only explicitly requested critic-only data (Eqs.
        24–27).
        """
        info = {}
        if self.critic_privileged:
            w = self.engine.world
            values = []
            from ..sim.ledger import inventory

            for i, state in enumerate(w.carriers):
                values.extend(state.capacity)
                values.extend(inventory(state, self.compiled, i))
            info["critic_state"] = np.asarray(values, dtype=np.float32)
            info["critic_only"] = True
        return info

    def action(self, action):
        """
        Convert a grid index or bounded vector to a typed carrier action (Eqs. 24–27).
        """
        if isinstance(action, (CoefficientAction, PriceAction, HoldAction)):
            return action
        if isinstance(action, (int, np.integer)):
            if not 0 <= int(action) < len(self.pricing.theta_grid or ()):
                raise IndexError("action index outside theta grid")
            return CoefficientAction(tuple(self.pricing.theta_grid[int(action)]))
        array = np.asarray(action, dtype=float)
        lo, hi = np.array(self.pricing.theta_bounds).T
        return CoefficientAction(tuple(np.clip(array, lo, hi)))

    def step(self, action):
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        24–27).
        """
        before = self.engine.world.clone()
        result = self.engine.step({self.i: self.action(action)})
        cash_reward = self.objective.period_reward(self.i, before, self.engine.world)
        terminal = self.engine.terminal()
        terminal_reward = (
            self.objective.terminal_reward(self.engine.world) if terminal else 0.0
        )
        reward = cash_reward + terminal_reward
        obs = self.engine.observe(self.i, self.arm)
        info = {
            **self._info(),
            "period_reward": cash_reward,
            "terminal_reward": terminal_reward,
            "posted_prices": tuple(self.engine.world.carriers[self.i].previous_prices),
            "own_outcome": obs.own_purchases,
        }
        return obs, float(reward), terminal, info

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 24–27).
        """
        return self.engine.snapshot()

    def restore(self, s):
        """Restore a previously captured execution/filter snapshot (Eqs. 24–27)."""
        self.engine.restore(s)

    def branch(self):
        """
        Clone mutable state while sharing immutable model data and random tapes (Eqs.
        24–27).
        """
        result = copy.copy(self)
        result.engine = self.engine.branch()
        return result
