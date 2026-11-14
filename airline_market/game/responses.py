"""Best-response policy factories; certification names the response class."""

from ..core.types import CoefficientAction
from ..pomdp.env import FocalCarrierEnv
from ..belief import ExactBelief
from ..solvers import ExactBeliefDP, ScenarioMPC, MPCConfig
from ..solvers.fixed_search import search_fixed_grid
from ..core.rng import StreamRegistry
from .payoffs import PolicyRef, controller


class BeliefPlannerController:
    """
    History-conditioned coefficient controller backed by a belief planner (Eqs. 20–23).
    """

    policy_class = "coefficient_history"
    exact_compatible = False

    def __init__(self, kind, arm, rivals, params):
        """
        Bind declared inputs and initialize BeliefPlannerController bookkeeping (Eqs.
        20–23).
        """
        self.kind = kind
        self.arm_name = arm
        self.rivals = rivals
        self.params = params

    def reset(self, ctx, rng):
        """Start a new cohort and clear per-episode memory (Eqs. 20–23)."""
        self.ctx = ctx
        self.code = ctx.compiled.spec.carriers[ctx.carrier].code
        self.before = None
        self.previous_action = None
        rivals = {i: controller(ref, i, 0) for i, ref in self.rivals.items()}
        from ..sim.kernel import ExactKernel

        kernel = ExactKernel(ctx.compiled, rivals)
        self.belief = ExactBelief(ctx.compiled, self.code, self.arm_name, kernel)
        if self.kind == "exact_dp":
            self.planner = ExactBeliefDP(
                ctx.compiled, self.code, self.arm_name, rivals, **self.params
            )
            self.planner.solve()
        else:
            self.planner = ScenarioMPC(
                ctx.compiled,
                self.code,
                self.arm_name,
                self.belief,
                rivals,
                config=MPCConfig(**self.params),
                plan_streams=StreamRegistry(int(rng.integers(2**31)), "plan"),
            )

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        20–23).
        """
        if self.before is None:
            self.belief.reset(obs)
        else:
            self.belief.update(self.before, self.previous_action, obs)
        if self.kind == "exact_dp":
            action = CoefficientAction(
                self.planner.grid[self.planner.policy()(obs, self.belief)]
            )
        else:
            action = self.planner.choose(obs, 0).action
        self.before = obs
        self.previous_action = action
        return action

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
        return self.before, self.previous_action, self.belief.snapshot()

    def restore(self, value):
        """Restore a previously captured execution/filter snapshot (Eqs. 20–23)."""
        self.before, self.previous_action = value[:2]
        self.belief.restore(value[2])


def best_response_factory(spec, solver, seeds, *, arms=None, params=None):
    """
    Construct a response solver for the specified policy class and seed split (Eqs.
    20–23).
    """
    from ..spec import compile_market

    c = compile_market(spec)
    params = params or {}
    arms = arms or {
        i.code: next((a.name for a in spec.information if a.observer == i.code), "own")
        for i in spec.carriers
    }

    def best_response(carrier, profile):
        """
        Choose or construct a response to the frozen opponent profile (Eqs. 20–23).
        """
        rivals = {i: p for i, p in profile.items() if i != carrier}
        arm = arms[carrier]
        if solver == "fixed_grid":
            factory = lambda: FocalCarrierEnv(
                c, carrier, arm, {i: controller(r, i, 0) for i, r in rivals.items()}
            )
            grid = next(p.theta_grid for p in spec.pricing if p.carrier == carrier)
            result = search_fixed_grid(factory, grid, seeds)
            return PolicyRef("fixed", result.best_theta, arm)
        if solver in ("exact_dp", "mpc_policy"):
            return PolicyRef(solver, {"rivals": rivals, "params": params}, arm)
        if solver == "ppo":
            from ..learn.ppo import train_ppo
            from ..core.hashing import jsonable

            config = {
                **params["train_config"],
                "spec": jsonable(spec),
                "focal": carrier,
                "arm": arm,
                "rival_controllers": {i: jsonable(ref) for i, ref in rivals.items()},
            }
            return PolicyRef("checkpoint", train_ppo(config).checkpoint, arm)
        raise ValueError("unknown best response solver " + solver)

    return best_response
