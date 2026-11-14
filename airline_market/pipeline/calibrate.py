from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..spec import apply_overrides, compile_market
from ..sim.engine import MarketEngine
from ..agents.fixed import FixedCoefficientController
from ..vendor.ride_response.cem import CEMOptimizer, CEMConfig
from ..core.rng import StreamRegistry


@dataclass(frozen=True)
class MomentTarget:
    """
    Observed calibration statistic with declared scale, weight and tolerance (model
    §§1–3, calibration assumptions).
    """

    name: str
    value: float
    weight: float = 1.0
    tolerance: float = 0.05
    statistic: object = None


@dataclass
class CalibrationResult:
    """
    Fitted model, paired moment diagnostics and identification limitations (model §§1–3,
    calibration assumptions).
    """

    params: dict
    spec: object
    objective: float
    fit_table: list
    seeds: tuple
    warnings: list


class SimulatedMomentCalibrator:
    """
    Common-tape simulated-moment parameter fitting and model ensembles (model §§1–3,
    calibration assumptions).
    """

    def __init__(
        self,
        base_spec,
        parameter_space,
        targets,
        baseline_controllers=None,
        seeds=range(10),
        cem=None,
    ):
        """
        Bind declared inputs and initialize SimulatedMomentCalibrator bookkeeping (model
        §§1–3, calibration assumptions).
        """
        self.base_spec = base_spec
        self.space = parameter_space
        self.targets = targets
        self.controllers = baseline_controllers or {}
        self.seeds = tuple(seeds)
        self.cem = cem or CEMConfig()
        self.evaluated = []

    def evaluate(self, params):
        """
        Evaluate the supplied model candidate under the declared objective (model §§1–3,
        calibration assumptions).
        """
        spec = apply_overrides(self.base_spec, params)
        c = compile_market(spec)
        controllers = dict(self.controllers)
        for a in spec.controllers:
            if a.controller == "external" and a.carrier not in controllers:
                p = next(p for p in spec.pricing if p.carrier == a.carrier)
                controllers[a.carrier] = FixedCoefficientController(
                    tuple(
                        np.clip(
                            np.zeros(len(p.theta_bounds)), *np.asarray(p.theta_bounds).T
                        )
                    )
                )
        rows = []
        for seed in self.seeds:
            import copy

            e = MarketEngine(c, copy.deepcopy(controllers))
            e.reset(seed)
            while not e.terminal():
                e.step()
            rows.append(e.metrics())
        table = []
        for target in self.targets:
            predicted = float(
                target.statistic(rows)
                if target.statistic
                else registered_moment(target.name, rows, c)
            )
            error = (predicted - target.value) / max(
                abs(target.value), target.tolerance, 1e-9
            )
            table.append(
                {
                    "name": target.name,
                    "target": target.value,
                    "predicted": predicted,
                    "relative_error": error,
                    "weight": target.weight,
                    "within_tolerance": abs(predicted - target.value)
                    <= target.tolerance,
                }
            )
        value = sum(r["weight"] * r["relative_error"] ** 2 for r in table)
        self.evaluated.append((value, dict(params), spec))
        return value, table

    def fit(self):
        """
        Fit declared parameters on fixed calibration tapes and flag flat directions
        (model §§1–3, calibration assumptions).
        """
        names = list(self.space)
        bounds = np.array(list(self.space.values()))
        optimizer = CEMOptimizer(len(names), self.cem)

        def objective(x):
            """
            Score one candidate on the same training/calibration random inputs (model
            §§1–3, calibration assumptions).
            """
            return -self.evaluate(dict(zip(names, np.clip(x, *bounds.T))))[0]

        result = optimizer.optimize(
            objective,
            initial_mean=bounds.mean(1),
            initial_std=np.maximum(np.ptp(bounds, axis=1) / 3, 0.001),
        )
        params = dict(zip(names, np.clip(result["coordinates"], *bounds.T)))
        value, table = self.evaluate(params)
        warnings = []
        for k, name in enumerate(names):
            scores = []
            for x in np.linspace(*bounds[k], 5):
                scores.append(self.evaluate({**params, name: float(x)})[0])
            if np.ptp(scores) < 1e-4:
                warnings.append("weakly identified/flat objective along " + name)
        return CalibrationResult(
            params,
            apply_overrides(self.base_spec, params),
            value,
            table,
            self.seeds,
            warnings,
        )

    def ensemble(self, n, method="cem_elites"):
        """
        Return a weighted set of accepted or elite calibrated specifications (model
        §§1–3, calibration assumptions).
        """
        if not self.evaluated:
            self.fit()
        if method == "cem_elites":
            chosen = sorted(self.evaluated, key=lambda x: x[0])[:n]
        elif method == "abc_rejection":
            rng = StreamRegistry(self.cem.seed, "calibration/abc").generator()
            threshold = sorted(x[0] for x in self.evaluated)[
                min(n - 1, len(self.evaluated) - 1)
            ]
            chosen = []
            for _ in range(n * 100):
                params = {k: float(rng.uniform(*b)) for k, b in self.space.items()}
                v, _ = self.evaluate(params)
                if v <= threshold:
                    chosen.append(self.evaluated[-1])
                if len(chosen) >= n:
                    break
        else:
            raise ValueError("unknown ensemble method")
        return [(spec, 1 / len(chosen)) for _, _, spec in chosen]


def calibrate_scope(name, catalog, out, ensemble):
    """
    Compile a local-data scope and fit its declared calibration parameters (model §§1–3,
    calibration assumptions).
    """
    import json
    from .catalog import load_scope
    from .compiler import compile_scope

    scope = load_scope(name, catalog)
    spec = compile_scope(scope)
    target_path = scope.params.get("targets_file")
    targets = (
        [MomentTarget(**t) for t in json.loads(Path(target_path).read_text())]
        if target_path
        else default_targets(scope, spec)
    )
    space = scope.params.get(
        "parameter_space",
        {
            "segments.0.classes.0.alpha": (0.001, 0.04),
            "segments.1.classes.0.alpha": (0.001, 0.025),
        },
    )
    fit = SimulatedMomentCalibrator(
        spec, space, targets, seeds=range(scope.params.get("calibration_cohorts", 16))
    )
    result = fit.fit()
    from .artifacts import save_calibration

    dest = save_calibration(name, result, root=out)
    return {"output": str(dest), "fit": result, "ensemble": fit.ensemble(ensemble)}


def registered_moment(name, rows, compiled):
    """
    Compute a named simulation moment using its declared aggregation (model §§1–3,
    calibration assumptions).
    """
    if "/" not in name:
        return float(np.mean([r[name] for r in rows]))
    fields = name.split("/")
    kind = fields[0]
    if kind == "load_factor":
        leg = compiled.leg_index[fields[1]]
        carrier = int(compiled.leg_carrier[leg])
        code = compiled.spec.carriers[carrier].code
        local = compiled.legs_of[carrier].index(leg)
        return float(np.mean([r["carriers"][code]["load_factor"][local] for r in rows]))
    if kind == "nonstop_share":
        market = fields[1]
        values = [r["markets"][market]["carriers"] for r in rows]
        num = sum(sum(v["nonstop_journeys"] for v in x.values()) for x in values)
        den = sum(sum(v["journeys"] for v in x.values()) for x in values)
        return num / max(1, den)
    carrier, market = fields[1:]
    values = [r["markets"][market]["carriers"] for r in rows]
    if kind == "carrier_share":
        return sum(x[carrier]["journeys"] for x in values) / max(
            1, sum(sum(v["journeys"] for v in x.values()) for x in values)
        )
    if kind in ("median_fare", "fare_dispersion"):
        observed = [
            (x[carrier][kind], x[carrier]["journeys"])
            for x in values
            if x[carrier][kind] is not None
        ]
        # Cohort moment approximation is explicit; pooled booking quantiles
        # require keeping individual transactions from the trace.
        return (
            float(
                np.average([x for x, n in observed], weights=[n for x, n in observed])
            )
            if observed
            else 0.0
        )
    raise ValueError("unknown calibration moment " + name)


def default_targets(scope, spec):
    """
    Build observed-moment targets from available T-100 and DB1B subsets (model §§1–3,
    calibration assumptions).
    """
    from .sources import t100, db1b

    capacity = t100.load_segments(scope)
    fares = db1b.load_markets(scope)
    targets = []
    for leg in spec.legs:
        subset = capacity[
            (capacity.carrier == leg.carrier)
            & (capacity.origin == leg.origin)
            & (capacity.dest == leg.dest)
        ]
        if len(subset):
            targets.append(
                MomentTarget(
                    "load_factor/" + leg.id,
                    float(subset.load_factor.mean()),
                    tolerance=0.05,
                )
            )
    for market in spec.markets:
        subset = fares[(fares.origin == market.origin) & (fares.dest == market.dest)]
        total = float(subset.passengers.sum())
        if not total:
            continue
        targets.append(
            MomentTarget(
                "nonstop_share/" + market.id,
                float(subset[subset.coupons == 1].passengers.sum() / total),
                tolerance=0.05,
            )
        )
        for carrier in spec.carriers:
            f = subset[subset.carrier == carrier.code]
            if f.empty:
                continue
            targets.extend(
                [
                    MomentTarget(
                        "carrier_share/" + carrier.code + "/" + market.id,
                        float(f.passengers.sum() / total),
                        tolerance=0.05,
                    ),
                    MomentTarget(
                        "median_fare/" + carrier.code + "/" + market.id,
                        float(f.p50.median()),
                        tolerance=10.0,
                    ),
                ]
            )
    return targets
