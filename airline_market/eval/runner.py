"""Train-only selection followed by paired test evaluation."""

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import copy, json, time
from ..spec import load_spec, compile_market, apply_overrides
from ..core.hashing import spec_hash, config_hash, jsonable
from ..core.types import CoefficientAction
from ..core.rng import StreamRegistry
from ..pomdp.env import FocalCarrierEnv
from ..belief import ExactBelief, ParticleBelief, FactorizedBelief, PointBelief
from ..sim.tape import LegacyToyTapeAdapter, ExogenousTape
from ..legacy_toy.model import NoiseTape
from ..solvers import (
    search_fixed_grid,
    search_schedule,
    ExactBeliefDP,
    ScenarioMPC,
    MPCConfig,
    LegacyRolloutPlanner,
)
from ..solvers.legacy_rollout import planning_seed
from ..vendor.ride_response.cem import CEMConfig
from .seeds import SeedPlan
from .reports import statistics, paired_statistics, save_results


class ExperimentResult(dict):
    """Serializable paired experiment outcomes and provenance (Eqs. 20–22, 29)."""

    pass


def reset(env, seed, tape=None, legacy=False):
    """Start a new cohort and clear per-episode memory (Eqs. 20–22, 29)."""
    if legacy:
        tape = LegacyToyTapeAdapter.from_legacy(
            NoiseTape.from_seed(seed, env.compiled.spec.horizon.n_periods), env.compiled
        )
    if tape is None:
        return env.reset(seed)[0]
    env.engine.reset(seed, tape=tape)
    return env.engine.observe(env.i, env.arm)


def _evaluate_seed(job):
    """Evaluate every method on one shared, pre-indexed cohort tape (Eqs. 20–22, 29)."""
    spec, config, selected, seed, index = job
    c = compile_market(load_spec(spec))
    focal = config.get("focal", c.spec.carriers[0].code)
    master = config["seeds"].get("planning_master", 773)
    tape = ExogenousTape.generate(c, seed)
    legacy = config.get("legacy_tapes", False)
    rows = {}
    legacy_planner = None
    for method in config["methods"]:
        arm = method.get("arm", "direct")
        solver = method["solver"]
        params = dict(method.get("params", {}))
        env = FocalCarrierEnv(c, focal, arm)
        obs = reset(env, seed, tape, legacy)
        decisions = []
        b = None
        dp = None
        planner = None
        if solver in ("mpc", "legacy_rollout", "exact_dp"):
            kind = method.get("belief", "exact")
            cls = {
                "exact": ExactBelief,
                "particle": ParticleBelief,
                "factorized": FactorizedBelief,
                "point": PointBelief,
            }[kind]
            kwargs = {"seed": master + index} if kind == "particle" else {}
            b = cls(c, focal, arm, **kwargs)
            b.reset(obs)
        if solver == "legacy_rollout":
            if legacy_planner is None:
                legacy_planner = LegacyRolloutPlanner(
                    c, selected["fixed"], params.get("n_scenarios", 32)
                )
            planner = legacy_planner
        elif solver == "mpc":
            planner = ScenarioMPC(
                c,
                focal,
                arm,
                b,
                config=MPCConfig(**params),
                plan_streams=StreamRegistry(master, "plan"),
                tail_coefficients=selected["fixed"],
            )
        elif solver == "exact_dp":
            dp = ExactBeliefDP(c, focal, arm, **params)
            dp.solve()
        while obs.t < c.spec.horizon.n_periods:
            if solver == "fixed_grid":
                action = CoefficientAction(tuple(selected[method["name"]]))
            elif solver == "schedule_cem":
                sch = selected[method["name"]]
                action = CoefficientAction(
                    tuple(
                        sch["thetas"][
                            next(
                                k
                                for k, (a, z) in enumerate(sch["blocks"])
                                if a <= obs.t < z
                            )
                        ]
                    )
                )
            elif solver == "legacy_rollout":
                action = CoefficientAction(
                    planner.choose(
                        obs, b, planning_seed(master, index, obs.t)
                    ).coefficients
                )
            elif solver == "mpc":
                action = planner.choose(obs, index).action
            elif solver == "exact_dp":
                action = CoefficientAction(dp.grid[dp.policy()(obs, b)])
            else:
                raise ValueError("unknown experiment solver " + solver)
            if config.get("trace"):
                decisions.append(
                    {
                        "period": obs.t,
                        "coefficients": action.theta,
                        "belief_entropy": b.entropy() if b else None,
                    }
                )
            after, _, _, _ = env.step(action)
            if b is not None:
                b.update(obs, action, after)
            obs = after
        metrics = env.engine.metrics()
        if focal != c.spec.metadata.get("focal", c.spec.carriers[0].code):
            metrics.update(metrics["carriers"][focal])
        rows[method["name"]] = {"seed": seed, **metrics, "decisions": decisions}
    return rows


class ExperimentRunner:
    """
    Train/validate/select first, then evaluate on disjoint common test tapes (Eqs.
    20–22, 29).
    """

    def __init__(self, config, *, phase="development"):
        """
        Bind declared inputs and initialize ExperimentRunner bookkeeping (Eqs. 20–22,
        29).
        """
        self.config = (
            json.loads(Path(config).read_text())
            if isinstance(config, (str, Path))
            else copy.deepcopy(config)
        )
        if self.config.get("schema") != "airline-experiment-config/1":
            raise ValueError("unknown experiment schema")
        self.seeds = SeedPlan.from_config(self.config["seeds"])
        from .sealing import validate_seed_release

        self.phase = phase
        validate_seed_release(self.seeds, phase)

    def run(self, out=None):
        """
        Execute the declared paired experiment or backtest and return its outcomes (Eqs.
        20–22, 29).
        """
        started = time.monotonic()
        from .sealing import validate_seed_release

        validate_seed_release(self.seeds, self.phase)
        cfg = self.config
        spec = load_spec(cfg["spec"])
        spec = apply_overrides(spec, cfg.get("overrides", {}))
        c = compile_market(spec)
        focal = cfg.get("focal", spec.carriers[0].code)
        from ..spec.features import assert_arm_invariant_basis

        pricing = next(p for p in spec.pricing if p.carrier == focal)
        assert_arm_invariant_basis(
            pricing,
            [
                next(
                    a
                    for a in spec.information
                    if a.observer == focal and a.name == m.get("arm", "direct")
                )
                for m in cfg["methods"]
            ],
        )

        class TrainEnv(FocalCarrierEnv):
            """
            Training environment preserving the declared legacy tape adapter when
            requested (Eqs. 20–22, 29).
            """

            def reset(inner, seed):
                """Start a new cohort and clear per-episode memory (Eqs. 20–22, 29)."""
                if cfg.get("legacy_tapes"):
                    return reset(inner, seed, legacy=True), {}
                return super().reset(seed)

        factory = lambda: TrainEnv(c, focal, cfg["methods"][0].get("arm", "direct"))
        fixed = search_fixed_grid(factory, pricing.theta_grid, self.seeds.train)
        selected = {"fixed": fixed.best_theta}
        diagnostics = {"training_grid": fixed.table}
        for method in cfg["methods"]:
            if method["solver"] == "fixed_grid":
                selected[method["name"]] = fixed.best_theta
            elif method["solver"] == "schedule_cem":
                params = method.get("params", {})
                result = search_schedule(
                    factory,
                    pricing.theta_bounds,
                    self.seeds.train,
                    params["blocks"],
                    CEMConfig(**params.get("cem", {})),
                )
                selected[method["name"]] = result
                if self.seeds.validation:
                    diagnostics[method["name"] + "_validation"] = (
                        self._validate_schedule(factory, result)
                    )
            elif method["solver"] == "exact_dp":
                diagnostics[method["name"]] = {
                    "exact_value": ExactBeliefDP(
                        c, focal, method["arm"], **method.get("params", {})
                    )
                    .solve()
                    .value
                }
        opened = datetime.now(timezone.utc).isoformat()
        jobs = [
            (jsonable(spec), cfg, selected, seed, k)
            for k, seed in enumerate(self.seeds.test)
        ]
        workers = cfg.get("workers", 1)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_evaluate_seed, jobs))
        else:
            results = list(map(_evaluate_seed, jobs))
        episodes = {
            m["name"]: sorted([r[m["name"]] for r in results], key=lambda x: x["seed"])
            for m in cfg["methods"]
        }
        metrics = cfg.get(
            "metrics",
            [
                "profit",
                "share",
                "objective",
                "journeys",
                "refunds",
                "cancellations",
                "consumer_surplus",
            ],
        )
        summary = {
            m: {metric: statistics([r[metric] for r in rows]) for metric in metrics}
            for m, rows in episodes.items()
        }
        contrasts = {
            a + "_minus_" + b: {
                metric: paired_statistics(episodes[a], episodes[b], metric)
                for metric in metrics
            }
            for a, b in cfg.get("contrasts", [])
        }
        checks = {}
        for a, b in cfg.get("expect_identical", []):
            checks[a + "==" + b] = all(
                all(x[m] == y[m] for m in metrics)
                for x, y in zip(episodes[a], episodes[b])
            )
            if not checks[a + "==" + b]:
                raise AssertionError(f"expected identical: {a}, {b}")
        result = ExperimentResult(
            schema="airline-experiment/2",
            config=cfg,
            spec_hash=spec_hash(spec),
            config_hash=config_hash(cfg),
            seeds=asdict(self.seeds),
            test_opened_at=opened,
            selected=selected,
            diagnostics=diagnostics,
            summary=summary,
            contrasts=contrasts,
            episodes=episodes,
            identity_checks=checks,
            elapsed_seconds=time.monotonic() - started,
            scope="Synthetic finite departure-cohort pricing; coefficient policy comparisons. No calibrated airline, general equilibrium, or uncomputed exact-value claim.",
            phase=self.phase,
            planning_crn=True,
        )
        if out:
            save_results(result, out)
        return result

    def _validate_schedule(self, factory, schedule):
        """
        Evaluate a selected schedule on held-out validation seeds (Eqs. 20–22, 29).
        """
        from ..solvers.fixed_search import episode_value

        return [
            episode_value(
                factory(),
                s,
                lambda o: CoefficientAction(
                    tuple(
                        schedule["thetas"][
                            next(
                                k
                                for k, (a, b) in enumerate(schedule["blocks"])
                                if a <= o.t < b
                            )
                        ]
                    )
                ),
            )
            for s in self.seeds.validation
        ]
