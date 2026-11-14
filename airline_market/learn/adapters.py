"""Observation contract for public actors and privileged training critics."""

import numpy as np
from ..pomdp.observation import Vectorizer
from ..core.hashing import schema_hash
from ..belief import ExactBelief
from ..vendor.ride_response.recurrent.common import ActionTransform


class RecurrentEnvAdapter:
    """
    Stable public actor features plus separately masked critic-only data (Eqs. 18–19,
    26–27).
    """

    def __init__(
        self,
        env,
        vectorizer=None,
        action_transform=None,
        *,
        belief_features=False,
        joint_events=(),
    ):
        """
        Bind declared inputs and initialize RecurrentEnvAdapter bookkeeping (Eqs. 18–19,
        26–27).
        """
        self.env = env
        self.vectorizer = vectorizer or Vectorizer(env.compiled, env.arm)
        bounds = np.asarray(env.pricing.theta_bounds)
        self.action_transform = action_transform or ActionTransform(
            tuple("theta/" + str(k) for k in range(len(bounds))),
            ("bounded",) * len(bounds),
            tuple(bounds[:, 0]),
            tuple(bounds[:, 1]),
        )
        self.belief_features = belief_features
        self.events = joint_events
        self.belief = (
            ExactBelief(env.compiled, env.focal, env.arm.name)
            if belief_features
            else None
        )
        self.policy_class = (
            "belief_features" if belief_features else "coefficient_recurrent"
        )
        self._names = list(self.vectorizer.names())
        self.public_dim = len(self._names)
        if belief_features:
            # Fixed-width mean/zero/one inventory probabilities plus joint events.
            for i, carrier in enumerate(env.compiled.spec.carriers):
                if i != env.i:
                    for l in env.compiled.legs_of[i]:
                        self._names.extend(
                            f"belief/{env.compiled.spec.legs[l].id}/{x}"
                            for x in ["mean", "p0", "p1"]
                        )
            self._names.extend(
                "belief/event/" + str(k) for k in range(len(joint_events))
            )
        self.public_dim = len(self._names)
        if env.critic_privileged:
            self._names.extend(
                "critic/" + str(k) for k in range(2 * len(env.compiled.spec.legs))
            )

    def names(self):
        """
        Return stable feature names in the encoded observation order (Eqs. 18–19,
        26–27).
        """
        return tuple(self._names)

    def schema_hash(self):
        """Fingerprint the exact observation feature contract (Eqs. 18–19, 26–27)."""
        return schema_hash((self.names(), self.policy_class))

    def encode(self, obs, info):
        """
        Encode public/belief actor features, followed only by flagged critic inputs
        (Eqs. 18–19, 26–27).
        """
        public = list(self.vectorizer(obs))
        if self.belief:
            for _, p in sorted(self.belief.leg_marginals().items()):
                public.extend(
                    [
                        float(np.arange(len(p)) @ p),
                        float(p[0]),
                        float(p[1]) if len(p) > 1 else 0.0,
                    ]
                )
            public.extend(self.belief.event_probability(event) for event in self.events)
        critic = list(info["critic_state"]) if self.env.critic_privileged else []
        return np.asarray(public + critic, dtype=np.float32)

    def reset(self, seed):
        """Start a new cohort and clear per-episode memory (Eqs. 18–19, 26–27)."""
        self.obs, info = self.env.reset(seed)
        if self.belief:
            self.belief.reset(self.obs)
        return self.encode(self.obs, info)

    def step(self, action):
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        18–19, 26–27).
        """
        before = self.obs
        after, r, done, info = self.env.step(action)
        if self.belief:
            self.belief.update(before, self.env.action(action), after)
        self.obs = after
        return self.encode(after, info), r, done, info

    def configure_actor(self, agent):
        """
        Mask critic-only coordinates on every recurrent actor forward path (Eqs. 18–19,
        26–27).
        """
        agent.actor.input_mask[self.public_dim : len(self.names())] = 0.0
