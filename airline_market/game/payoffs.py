"""Paired payoff estimates cached by specification, policies and cohort seed."""

from dataclasses import dataclass
import copy
import numpy as np
from ..core.hashing import spec_hash
from ..core.rng import StreamRegistry
from ..spec import compile_market
from ..spec.model import ControllerAssignment
from ..agents import make_controller
from ..agents.fixed import FixedCoefficientController
from ..sim.engine import MarketEngine
from ..sim.accounting import carrier_metrics
from ..eval.reports import statistics


@dataclass(frozen=True)
class PolicyRef:
    """
    Serializable fixed/controller/checkpoint/mixture policy reference (Eqs. 20–23).
    """

    kind: str
    value: object
    arm: str | None = None

    def hash(self):
        """
        Fingerprint policy parameters and checkpoint contents for payoff caching (Eqs.
        20–23).
        """
        if self.kind == "checkpoint":
            import hashlib
            from pathlib import Path

            return spec_hash(
                (self, hashlib.sha256(Path(self.value).read_bytes()).hexdigest())
            )
        return spec_hash(self)


@dataclass
class PayoffEstimate:
    """Carrier objectives and paired per-seed payoff observations (Eqs. 20–23)."""

    means: dict
    per_seed: dict
    seeds: tuple
    statistics: dict


PAYOFF_CACHE = {}


def controller(ref, carrier, seed):
    """Resolve a policy reference to its executable controller (Eqs. 20–23)."""
    if ref.kind == "fixed":
        return FixedCoefficientController(tuple(ref.value))
    if ref.kind == "controller":
        return make_controller(ControllerAssignment(carrier=carrier, **ref.value))
    if ref.kind == "mixed":
        return MixedController(ref.value, carrier)
    if ref.kind == "checkpoint":
        return CheckpointController(ref.value, ref.arm)
    if ref.kind in ("exact_dp", "mpc_policy"):
        from .responses import BeliefPlannerController

        return BeliefPlannerController(
            ref.kind, ref.arm, ref.value["rivals"], ref.value.get("params", {})
        )
    raise ValueError("unknown policy reference " + ref.kind)


class CheckpointController:
    """Deterministic execution of a schema-checked recurrent checkpoint (Eqs. 20–23)."""

    policy_class = "coefficient_recurrent"
    exact_compatible = False

    def __init__(self, path, arm):
        """
        Bind declared inputs and initialize CheckpointController bookkeeping (Eqs.
        20–23).
        """
        import torch

        self.path = path
        self.arm_name = arm or torch.load(path, weights_only=False)["metadata"]["arm"]

    def reset(self, ctx, rng):
        """Start a new cohort and clear per-episode memory (Eqs. 20–23)."""
        from ..learn.adapters import RecurrentEnvAdapter
        from ..learn.ppo import load_checkpoint
        from ..pomdp.env import FocalCarrierEnv
        import torch

        payload = torch.load(self.path, weights_only=False)
        meta = payload["metadata"]
        self.adapter = RecurrentEnvAdapter(
            FocalCarrierEnv(
                ctx.compiled,
                ctx.compiled.spec.carriers[ctx.carrier].code,
                self.arm_name or meta["arm"],
                critic_privileged=any(
                    n.startswith("critic/") for n in meta["observation_names"]
                ),
            ),
            belief_features=meta["policy_class"] == "belief_features",
        )
        self.agent = load_checkpoint(self.path, self.adapter)
        self.hidden = None
        self.previous = None
        self.previous_coordinates = None
        self.previous_cash = 0.0
        self.previous_obs = None
        self.ctx = ctx

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        20–23).
        """
        from ..core.types import CoefficientAction

        if self.adapter.belief_features:
            if self.previous_obs is None:
                self.adapter.belief.reset(obs)
            else:
                self.adapter.belief.update(
                    self.previous_obs, CoefficientAction(tuple(self.previous)), obs
                )
        info = {
            "critic_state": np.zeros(
                len(self.adapter.names()) - self.adapter.public_dim
            )
        }
        x = self.adapter.encode(obs, info)
        action, self.hidden, policy_info = self.agent.act(
            x,
            hidden=self.hidden,
            previous_action=self.previous,
            previous_coordinates=self.previous_coordinates,
            previous_reward=(obs.own_cash - self.previous_cash)
            / self.ctx.objective.profit_scale,
            deterministic=True,
        )
        self.previous = action
        self.previous_coordinates = policy_info["policy_coordinates"]
        self.previous_cash = obs.own_cash
        self.previous_obs = obs
        return CoefficientAction(tuple(action))

    def observe_outcome(self, *args):
        """
        Update controller bookkeeping from its own realized period outcome (Eqs. 20–23).
        """
        pass

    def memory(self):
        """
        Return the controller's hashable finite memory, without world references (Eqs.
        20–23).
        """
        return ()

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 20–23).
        """
        return copy.deepcopy(
            (
                self.hidden,
                self.previous,
                self.previous_coordinates,
                self.previous_cash,
                self.previous_obs,
                self.adapter.belief.snapshot() if self.adapter.belief else None,
            )
        )

    def restore(self, memory):
        """Restore a previously captured execution/filter snapshot (Eqs. 20–23)."""
        (
            self.hidden,
            self.previous,
            self.previous_coordinates,
            self.previous_cash,
            self.previous_obs,
            belief,
        ) = copy.deepcopy(memory)
        if belief is not None:
            self.adapter.belief.restore(belief)


class MixedController:
    """
    Episode-level policy mixture drawn on the named carrier control stream (Eqs. 20–23).
    """

    policy_class = "episode_mixture"
    exact_compatible = False

    def __init__(self, value, carrier):
        """
        Bind declared inputs and initialize MixedController bookkeeping (Eqs. 20–23).
        """
        self.value = value
        self.carrier_code = carrier
        self.arm_name = None

    def reset(self, ctx, rng):
        """Start a new cohort and clear per-episode memory (Eqs. 20–23)."""
        self.rng = rng
        self.choice = int(
            rng.choice(len(self.value["policies"]), p=self.value["probabilities"])
        )
        ref = self.value["policies"][self.choice]
        ref = ref if isinstance(ref, PolicyRef) else PolicyRef(**ref)
        self.child = controller(ref, self.carrier_code, 0)
        self.child.reset(ctx, rng)
        self.arm_name = getattr(self.child, "arm_name", ref.arm)

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        20–23).
        """
        return self.child.act(obs)

    def observe_outcome(self, outcome):
        """
        Update controller bookkeeping from its own realized period outcome (Eqs. 20–23).
        """
        self.child.observe_outcome(outcome)

    def memory(self):
        """
        Return the controller's hashable finite memory, without world references (Eqs.
        20–23).
        """
        return self.choice, self.child.memory()

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 20–23).
        """
        return (
            self.choice,
            self.child.snapshot(),
            copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, value):
        """Restore a previously captured execution/filter snapshot (Eqs. 20–23)."""
        if value[0] != self.choice:
            raise ValueError(
                "cannot restore mixture snapshot into a different episode component"
            )
        self.child.restore(value[1])
        self.rng.bit_generator.state = copy.deepcopy(value[2])


def estimate_payoffs(spec, profile, seeds, *, cache=None):
    """
    Estimate per-carrier objectives on common tapes, caching policy/seed profiles (Eqs.
    20–23).
    """
    c = compile_market(spec)
    seeds = tuple(seeds)
    values = {i: [] for i in profile}
    store = PAYOFF_CACHE if cache is None else cache
    refs = {
        i: r if isinstance(r, PolicyRef) else PolicyRef(**r) for i, r in profile.items()
    }
    for seed in seeds:
        key = (c.spec_hash, tuple((i, r.hash()) for i, r in sorted(refs.items())), seed)
        if key not in store:
            e = MarketEngine(c, {i: controller(r, i, seed) for i, r in refs.items()})
            e.reset(seed)
            while not e.terminal():
                e.step()
            store[key] = {
                i: carrier_metrics(e.world, c, c.carrier_index[i])["objective"]
                for i in profile
            }
        for i in values:
            values[i].append(store[key][i])
    return PayoffEstimate(
        {i: float(np.mean(v)) for i, v in values.items()},
        values,
        seeds,
        {i: statistics(v) for i, v in values.items()},
    )
