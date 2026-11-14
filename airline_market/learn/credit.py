"""Network counterfactual returns on a training snapshot (R2).

Centered credit and hold-relative credit are distinct: only the latter is zero
for hold when the alternative set also contains actions with different returns.
"""

import numpy as np
from ..core.rng import StreamRegistry


def counterfactual_credit(
    env, alternatives, policy_factory, *, horizon=None, seed=0, hold_index=1
):
    """
    Branch one training tape and return centered and hold-relative returns (Eqs. 18–19,
    26–27).
    """
    returns = []
    T = env.compiled.spec.horizon.n_periods
    end = min(T, env.engine.world.t + (horizon or T))
    for k, action in enumerate(alternatives):
        branch = env.branch()
        policy = policy_factory()
        rng = StreamRegistry(seed, "train/cf").generator()
        obs, r, done, info = branch.step(action)
        total = r
        previous = action
        while not done and branch.engine.world.t < end:
            action = policy(obs, previous, r, rng)
            obs, r, done, info = branch.step(action)
            previous = action
            total += r
        returns.append(total)
    values = np.asarray(returns)
    return {
        "returns": values,
        "centered": values - values.mean(),
        "hold_relative": values - values[hold_index],
        "policy_class": "network_counterfactual_credit",
        "truncated_at": end,
    }
