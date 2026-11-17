import json

import numpy as np
import pytest

from airline_market.belief import ExactBelief, ParticleBelief
from airline_market.core.types import CoefficientAction
from airline_market.pipeline.generators import hub_feeder_family
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub


@pytest.mark.slow
def test_particle_50_episodes_two_networks(tmp_path):
    reports = []
    for spec in [toy_hub(), hub_feeder_family({"n_spokes": 2, "periods": 4})]:
        c = compile_market(spec)
        env = FocalCarrierEnv(c, "A", "linked")
        tv = []
        marginal = []
        for seed in range(50):
            obs, _ = env.reset(seed)
            b = ExactBelief(c, "A", "linked")
            p = ParticleBelief(c, "A", "linked", n_particles=20000, seed=seed + 500)
            b.reset(obs)
            p.reset(obs)
            while obs.t < spec.horizon.n_periods:
                action = CoefficientAction((0.0, 0.0, 0.0))
                after, _, _, _ = env.step(action)
                b.update(obs, action, after)
                p.update(obs, action, after)
                obs = after
                tv.append(
                    0.5
                    * sum(
                        abs(b.mass.get(h, 0.0) - p.mass.get(h, 0.0))
                        for h in set(b.mass) | set(p.mass)
                    )
                )
                for leg, arr in b.leg_marginals().items():
                    other = p.leg_marginals()[leg]
                    n = max(len(arr), len(other))
                    marginal.append(
                        float(
                            np.max(
                                abs(
                                    np.pad(arr, (0, n - len(arr)))
                                    - np.pad(other, (0, n - len(other)))
                                )
                            )
                        )
                    )
        reports.append(
            {
                "spec": spec.name,
                "spec_hash": c.spec_hash,
                "periods": spec.horizon.n_periods,
                "n_particles": 20000,
                "episodes": 50,
                "conditional_sampling": "systematic",
                "mean_tv": float(np.mean(tv)),
                "mean_max_leg_error": float(np.mean(marginal)),
                "max_leg_error": float(np.max(marginal)),
            }
        )
        assert np.mean(tv) <= 0.02
        assert np.max(marginal) <= 0.01
    (tmp_path / "particle_convergence.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
