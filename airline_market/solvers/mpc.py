"""Exact small-model rollout and protected common-scenario MPC (Eq. 26)."""

from collections import OrderedDict
from dataclasses import dataclass, field
import numpy as np
from ..core.types import CoefficientAction, PriceAction
from ..core.rng import StreamRegistry
from ..core.hashing import canonical_json
from ..sim.kernel import ExactKernel
from ..sim.tape import ExogenousTape
from ..sim.finite import FinitePricingModel, UnsupportedFiniteModel
from .finite_dp import ExactTailEvaluator
from ..vendor.ride_response.cem import CEMConfig, CEMOptimizer

_TAIL_CACHE = OrderedDict()


@dataclass(frozen=True)
class MPCConfig:
    """Rollout computation and finite-sample switching rule, declared in reports."""

    n_scenarios: int = 256
    lookahead: str = "one_step_then_tail"
    k: int = 1
    tail_policy: object = "fixed:best"
    selection: str = "all"
    candidates: str = "grid"
    cem: CEMConfig | None = None
    reuse_scenarios_across_periods: bool = False
    evaluation: str = "auto"
    protect_incumbent: bool = True
    switch_standard_errors: float = 2.0
    switch_absolute_margin: float = 0.0
    n_estimation_scenarios: int | None = None
    racing_initial_scenarios: int = 64

    def __post_init__(self):
        """
        Canonicalize declared fields and check the object's construction invariants
        (Eqs. 26–32).
        """
        if isinstance(self.cem, dict):
            object.__setattr__(self, "cem", CEMConfig(**self.cem))
        if self.n_scenarios < 1 or self.k < 1:
            raise ValueError("scenario count and k must be positive")
        if self.selection not in ("split", "all", "racing"):
            raise ValueError("unknown selection mode")
        if self.racing_initial_scenarios < 1:
            raise ValueError("racing initial sample must be positive")
        if self.selection == "racing" and (
            self.candidates != "grid" or self.lookahead == "open_loop_cem"
        ):
            raise ValueError(
                "successive-halving racing requires finite grid candidates"
            )
        if self.selection == "split" and self.n_scenarios < 2:
            raise ValueError("split selection requires at least two scenarios")
        if self.lookahead not in ("one_step_then_tail", "k_step_hold", "open_loop_cem"):
            raise ValueError("unknown lookahead")
        if self.evaluation not in ("auto", "exact", "sampling"):
            raise ValueError("unknown evaluation mode")
        if self.switch_standard_errors < 0 or self.switch_absolute_margin < 0:
            raise ValueError("switching margins must be nonnegative")
        if self.n_estimation_scenarios is not None and self.n_estimation_scenarios < 1:
            raise ValueError("estimation sample must be positive")


@dataclass(frozen=True)
class MPCDecision:
    """Selected coefficients, independently evaluated value and diagnostics."""

    action: CoefficientAction
    estimate: float
    std_error: float | None
    estimate_by_candidate: tuple
    n_scenarios: int
    diagnostics: dict = field(default_factory=dict)


class ScenarioMPC:
    """Use exact fixed-tail Q when feasible; otherwise protect the incumbent.

    In sampling mode selection uses all scenarios by default. A fresh planning
    batch reports the selected value, avoiding reuse of its winning noise.
    The paired switching margin is a heuristic, not a simultaneous guarantee.
    """

    def __init__(
        self,
        compiled,
        focal,
        arm,
        belief,
        rival_controllers=None,
        config=None,
        plan_streams=None,
        *,
        tail_coefficients=None,
    ):
        """Bind declared inputs and initialize ScenarioMPC bookkeeping (Eqs. 26–32)."""
        self.compiled = compiled
        self.focal = compiled.carrier_index[focal]
        self.arm = arm
        self.belief = belief
        self.config = config or MPCConfig()
        self.streams = plan_streams or StreamRegistry(773, "plan")
        self.kernel = ExactKernel(compiled, rival_controllers)
        self.pricing = next(p for p in compiled.spec.pricing if p.carrier == focal)
        self.tail = (
            tuple(tail_coefficients)
            if tail_coefficients is not None
            else tuple(np.mean(self.pricing.theta_bounds, axis=1))
        )
        self.last_decision = None
        self.exact_evaluator = None
        self.exact_unavailable_reason = None
        cfg = self.config
        exact_tail = (
            cfg.tail_policy == "fixed:best"
            or isinstance(cfg.tail_policy, dict)
            and "thetas" in cfg.tail_policy
        )
        small = (
            compiled.spec.horizon.n_periods <= 12
            and len(compiled.spec.products) <= 12
            and sum(max(l.capacity.support) for l in compiled.spec.legs) <= 16
        )
        if (
            cfg.evaluation != "sampling"
            and exact_tail
            and cfg.lookahead == "one_step_then_tail"
            and cfg.candidates == "grid"
            and (small or cfg.evaluation == "exact")
        ):
            try:
                model = FinitePricingModel(compiled, focal, arm, rival_controllers)
                tail = (
                    cfg.tail_policy if isinstance(cfg.tail_policy, dict) else self.tail
                )
                key = (
                    compiled.spec_hash,
                    focal,
                    arm,
                    canonical_json(model.rival_rules),
                    canonical_json(tail),
                )
                if key not in _TAIL_CACHE:
                    _TAIL_CACHE[key] = ExactTailEvaluator(model, tail)
                    if len(_TAIL_CACHE) > 16:
                        _TAIL_CACHE.popitem(last=False)
                self.exact_evaluator = _TAIL_CACHE[key]
            except UnsupportedFiniteModel as error:
                self.exact_unavailable_reason = str(error)
        if cfg.evaluation == "exact" and self.exact_evaluator is None:
            raise UnsupportedFiniteModel(
                self.exact_unavailable_reason
                or "exact rollout requires a fixed/scheduled tail and one-step grid candidates"
            )

    def _tail(self, obs):
        """Resolve the incumbent coefficients at the current departure period."""
        policy = self.config.tail_policy
        if callable(policy):
            return policy(obs)
        if isinstance(policy, dict) and "thetas" in policy:
            return CoefficientAction(
                tuple(
                    policy["thetas"][
                        next(
                            k
                            for k, (a, b) in enumerate(policy["blocks"])
                            if a <= obs.t < b
                        )
                    ]
                )
            )
        return CoefficientAction(self.tail)

    def _scenarios(self, obs, episode_index, count, stage):
        """Draw model hypotheses and forecast tapes from a distinct plan stream."""
        period = 0 if self.config.reuse_scenarios_across_periods else obs.t
        rng = self.streams.generator(episode_index, period, stage)
        initial = []
        for h in self.belief.sample(count, rng):
            e = self.kernel.engine(h.kernel, self.focal)
            e.tape = ExogenousTape.generate(self.compiled, int(rng.integers(2**63)))
            initial.append(e)
        return initial

    def _evaluate(self, theta, initial, obs, episode_index):
        """Execute one candidate with shared forecasts and endogenous reactions."""
        cfg = self.config
        schedule = np.asarray(theta).reshape(-1, len(self.pricing.theta_bounds))
        values = []
        for scenario, e0 in enumerate(initial):
            e = e0.branch()
            tail_controller = None
            policy = cfg.tail_policy
            if (isinstance(policy, str) and policy != "fixed:best") or (
                isinstance(policy, dict) and "controller" in policy
            ):
                from ..agents import make_controller
                from ..agents.base import ControllerContext
                from ..spec.model import ControllerAssignment

                code = self.compiled.spec.carriers[self.focal].code
                assignment = ControllerAssignment(
                    carrier=code,
                    controller=policy
                    if isinstance(policy, str)
                    else policy["controller"],
                    params={} if isinstance(policy, str) else policy.get("params", {}),
                )
                tail_controller = make_controller(assignment)
                tail_controller.reset(
                    ControllerContext(
                        self.compiled,
                        self.focal,
                        self.pricing,
                        next(
                            o
                            for o in self.compiled.spec.objectives
                            if o.carrier == code
                        ),
                        next(
                            a
                            for a in self.compiled.spec.information
                            if a.observer == code and a.name == self.arm
                        ),
                    ),
                    self.streams.generator(episode_index, obs.t, 909, scenario),
                )
            while not e.terminal():
                t = e.world.t - obs.t
                current = e.observe(self.focal, self.arm)
                if cfg.lookahead == "open_loop_cem" and t < len(schedule):
                    action = CoefficientAction(tuple(schedule[t]))
                elif t < (cfg.k if cfg.lookahead == "k_step_hold" else 1):
                    action = CoefficientAction(tuple(schedule[0]))
                else:
                    action = (
                        tail_controller.act(current)
                        if tail_controller
                        else self._tail(current)
                    )
                if t == 0 and isinstance(action, CoefficientAction):
                    old = e.world.carriers[self.focal].theta
                    theta0 = (
                        old
                        if old is not None and obs.t % self.pricing.update_every
                        else action.theta
                    )
                    prices = e.rules[self.focal].prices(theta0, obs)
                    e.world.carriers[self.focal].theta = theta0
                    action = PriceAction(
                        dict(zip(self.compiled.products_of[self.focal], prices))
                    )
                e.step({self.focal: action})
            from ..sim.accounting import carrier_metrics

            values.append(
                carrier_metrics(e.world, self.compiled, self.focal)["objective"]
            )
        return np.asarray(values)

    def _scores(self, candidates, initial, obs, episode_index):
        """Select the validated batch specialization when it supports this model."""
        from ..sim.batch import supports, rollout_values

        cfg = self.config
        if (
            cfg.lookahead != "open_loop_cem"
            and (
                cfg.tail_policy == "fixed:best"
                or isinstance(cfg.tail_policy, dict)
                and "thetas" in cfg.tail_policy
            )
            and supports(self.compiled, self.focal, initial)
        ):
            tail = cfg.tail_policy if isinstance(cfg.tail_policy, dict) else self.tail
            return rollout_values(
                self.compiled,
                self.focal,
                initial,
                candidates,
                tail,
                hold=cfg.k if cfg.lookahead == "k_step_hold" else 1,
            )
        return np.asarray(
            [self._evaluate(x, initial, obs, episode_index) for x in candidates]
        )

    def choose(self, obs, episode_index):
        """Choose by exact Q or paired improvement; estimate with fresh draws."""
        cfg = self.config
        candidates = list(self.pricing.theta_grid or ())
        incumbent = tuple(self._tail(obs).theta)
        if incumbent not in candidates:
            candidates.append(incumbent)
        incumbent_index = candidates.index(incumbent)
        if self.exact_evaluator is not None:
            q = self.exact_evaluator.q_values(obs, self.belief, candidates)
            best = int(np.argmax(q))
            result = MPCDecision(
                CoefficientAction(tuple(candidates[best])),
                float(q[best]),
                0.0,
                tuple(map(float, q)),
                0,
                {
                    "evaluation": "exact_tail",
                    "tail_states": self.exact_evaluator.states,
                    "incumbent_index": incumbent_index,
                    "selected_minus_incumbent": float(q[best] - q[incumbent_index]),
                    "candidate_policy_class": "coefficient",
                    "note": "exact fixed-tail rollout, not full-horizon optimal control",
                },
            )
            self.last_decision = result
            return result
        racing = None
        eligible = None
        if cfg.selection == "racing":
            a, eligible, racing = self._race(
                candidates, incumbent_index, obs, episode_index
            )
            cut = a.shape[1]
        else:
            initial = self._scenarios(obs, episode_index, cfg.n_scenarios, 0)
            cut = cfg.n_scenarios // 2 if cfg.selection == "split" else cfg.n_scenarios
        if cfg.selection == "racing":
            pass
        elif cfg.candidates == "cem" or cfg.lookahead == "open_loop_cem":
            bounds = np.asarray(self.pricing.theta_bounds)
            n = (
                self.compiled.spec.horizon.n_periods - obs.t
                if cfg.lookahead == "open_loop_cem"
                else 1
            )
            bounds = np.tile(bounds, (n, 1))
            candidates = [np.tile(incumbent, n)]
            scores = [self._evaluate(candidates[0], initial, obs, episode_index)]
            incumbent_index = 0

            def objective(x):
                """
                Score one candidate on the same training/calibration random inputs (Eqs.
                26–32).
                """
                theta = np.clip(x, *bounds.T)
                v = self._evaluate(theta, initial, obs, episode_index)
                candidates.append(theta)
                scores.append(v)
                return float(v[:cut].mean())

            CEMOptimizer(len(bounds), cfg.cem).optimize(
                objective,
                initial_mean=bounds.mean(1),
                initial_std=np.maximum(np.ptp(bounds, axis=1) / 3, 0.01),
            )
            a = np.asarray(scores)
        elif cfg.candidates == "grid":
            a = self._scores(candidates, initial, obs, episode_index)
        else:
            raise ValueError("unknown candidates mode")
        if cfg.selection != "racing" and not np.isfinite(a).all():
            raise FloatingPointError("nonfinite MPC scenario objective")
        selection = np.nanmean(a[:, :cut], axis=1)
        winner = (
            int(np.argmax(selection))
            if eligible is None
            else int(eligible[np.argmax(selection[eligible])])
        )
        differences = a[:, :cut] - a[incumbent_index, :cut]

        def se(x):
            """Use only evaluated paired scenarios for a racing candidate."""
            values = np.asarray(x)[np.isfinite(x)]
            return (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None
            )

        improvement = float(np.nanmean(differences[winner]))
        paired_se = se(differences[winner])
        threshold = (
            cfg.switch_absolute_margin + cfg.switch_standard_errors * paired_se
            if paired_se is not None
            else float("inf")
        )
        best = (
            winner
            if not cfg.protect_incumbent or improvement > threshold
            else incumbent_index
        )
        report_count = cfg.n_estimation_scenarios or cfg.n_scenarios
        reporting = self._scenarios(obs, episode_index, report_count, 1)
        held = self._scores(
            [candidates[best], candidates[incumbent_index]],
            reporting,
            obs,
            episode_index,
        )
        diagnostics = {
            "evaluation": "sampling",
            "selection": cfg.selection,
            "selection_estimate": float(selection[best]),
            "independent_estimate": float(held[0].mean()),
            "selection_bias_estimate": float(selection[best] - held[0].mean()),
            "candidate_difference_se": [se(x) for x in differences],
            "winner_minus_incumbent": improvement,
            "paired_improvement_se": paired_se,
            "switch_threshold": threshold if np.isfinite(threshold) else None,
            "incumbent_retained": best == incumbent_index,
            "protect_incumbent": cfg.protect_incumbent,
            "estimation_scenarios": report_count,
            "selected_minus_incumbent": float((held[0] - held[1]).mean()),
            "candidate_policy_class": "coefficient",
            "racing": racing,
            "note": "switching margin is a paired heuristic, not a simultaneous improvement certificate",
        }
        result = MPCDecision(
            CoefficientAction(
                tuple(np.asarray(candidates[best]).reshape(-1, len(self.tail))[0])
            ),
            float(held[0].mean()),
            se(held[0]),
            tuple(map(float, selection)),
            cfg.n_scenarios,
            diagnostics,
        )
        self.last_decision = result
        return result

    def _race(self, candidates, incumbent_index, obs, episode_index):
        """Double common samples for the top half, always retaining the incumbent.

        Round namespaces are independent of the reporting batch. Eliminated
        candidates retain their smaller sample sizes in diagnostics and cannot
        re-enter using stale means. The switching test uses the winner's full
        paired sample against the incumbent, not independent candidate draws.
        """
        cfg = self.config
        values = np.full((len(candidates), cfg.n_scenarios), np.nan)
        active = list(range(len(candidates)))
        rounds = []
        previous = 0
        size = min(cfg.racing_initial_scenarios, cfg.n_scenarios)
        evaluated = 0
        while True:
            fresh = self._scenarios(
                obs, episode_index, size - previous, 2 + len(rounds)
            )
            scores = self._scores(
                [candidates[k] for k in active], fresh, obs, episode_index
            )
            if not np.isfinite(scores).all():
                raise FloatingPointError("nonfinite racing scenario objective")
            values[np.asarray(active), previous:size] = scores
            evaluated += len(active) * (size - previous)
            rounds.append(
                {
                    "candidates": active.copy(),
                    "scenarios_per_survivor": size,
                    "additional_scenarios": size - previous,
                }
            )
            if size == cfg.n_scenarios:
                break
            means = np.nanmean(values[active, :size], axis=1)
            ranked = sorted(zip(active, means), key=lambda pair: (-pair[1], pair[0]))
            survivors = [k for k, _ in ranked[: max(1, (len(active) + 1) // 2)]]
            if incumbent_index not in survivors:
                survivors.append(incumbent_index)
            active = sorted(survivors)
            previous, size = size, min(2 * size, cfg.n_scenarios)
        return (
            values,
            active,
            {
                "rounds": rounds,
                "candidate_scenario_evaluations": evaluated,
                "samples_by_candidate": np.isfinite(values).sum(axis=1).tolist(),
                "eligible_final_candidates": active,
            },
        )
