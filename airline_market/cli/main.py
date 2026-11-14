"""Command-line entry points. Optional data/RL dependencies load on demand."""

import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone
from ..core.hashing import jsonable
from ..core.registry import append_manifest


def main(argv=None):
    """
    Parse CLI arguments and write the requested reproducible artifacts (model §10,
    reproducible execution).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "campaign":
        from ..campaign.cli import main as campaign_main

        return campaign_main(args[1:])
    if args and args[0] == "bench":
        from ..bench.cli import main as bench_main

        return bench_main(args[1:])
    if args[:2] == ["legacy", "experiment"]:
        from ..legacy_toy.experiment import main as legacy

        sys.argv = [sys.argv[0]] + args[2:]
        return legacy()
    p = argparse.ArgumentParser(prog="airline-market")
    p.add_argument(
        "group",
        choices=[
            "spec",
            "simulate",
            "solve",
            "experiment",
            "pipeline",
            "calibrate",
            "train",
            "game",
            "campaign",
            "bench",
        ],
    )
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    head = p.parse_args(args)
    q = argparse.ArgumentParser(prog="airline-market " + head.group)
    if head.group != "simulate":
        q.add_argument("operation" if head.group not in ("calibrate",) else "scope")
    q.add_argument("path", nargs="?")
    q.add_argument("--out")
    q.add_argument("--overrides", default="{}")
    q.add_argument("--focal", default="A")
    q.add_argument("--observer", default="A")
    q.add_argument("--arm", default="linked")
    q.add_argument(
        "--arms", nargs="+", default=["direct", "linked", "unrelated", "oracle"]
    )
    q.add_argument("--max-nodes", type=int, default=100000)
    q.add_argument("--lag", type=int, default=1)
    q.add_argument("--workers", type=int)
    q.add_argument("--controllers", default="{}")
    q.add_argument("--seeds", default="0:100")
    q.add_argument("--trace", action="store_true")
    q.add_argument("--omegas", default="0,.1,.25,.5,1")
    q.add_argument("--catalog", default="config/market_catalog.json")
    q.add_argument("--targets", default="default")
    q.add_argument("--ensemble", type=int, default=32)
    q.add_argument("--source")
    q.add_argument("--count-rows", action="store_true")
    q.add_argument("--inventory")
    q.add_argument("--phase", choices=["development", "final"], default="development")
    a = q.parse_args(head.arguments)
    operation = getattr(a, "operation", head.group)
    path = a.path
    spec_output = (
        Path(a.out)
        if head.group == "spec" and a.out and Path(a.out).suffix == ".json"
        else None
    )
    out = (
        spec_output.parent
        if spec_output
        else Path(
            a.out
            or (
                "results/"
                + head.group
                + "_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            )
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    from ..spec import load_spec, compile_market, dump_spec, apply_overrides

    if head.group == "spec":
        spec = apply_overrides(load_spec(path), json.loads(a.overrides))
        c = compile_market(spec)
        if operation == "derive-arms":
            from ..spec.information import add_derived_arms

            spec = add_derived_arms(c, a.observer, lag=a.lag)
        c = compile_market(spec)
        result = {"valid": True, "spec_hash": c.spec_hash}
        if operation == "compile":
            target = spec_output or out / "compiled.json"
            target.write_text(
                json.dumps(
                    {"schema": "airline-compiled-market/1", **jsonable(c)}, indent=2
                )
                + "\n"
            )
            result["compiled_file"] = str(target.resolve())
        else:
            dump_spec(spec, spec_output or out / "spec.json")
    elif head.group == "simulate":
        from ..sim.engine import MarketEngine
        from ..agents import make_controller
        from ..spec.model import ControllerAssignment

        c = compile_market(load_spec(path))
        controllers = {
            k: make_controller(ControllerAssignment(carrier=k, **v))
            for k, v in json.loads(a.controllers).items()
        }
        # A missing external assignment must be supplied explicitly by this CLI.
        from ..agents.fixed import FixedCoefficientController

        for assignment in c.spec.controllers:
            if (
                assignment.controller == "external"
                and assignment.carrier not in controllers
            ):
                pr = next(p for p in c.spec.pricing if p.carrier == assignment.carrier)
                controllers[assignment.carrier] = FixedCoefficientController(
                    pr.theta_grid[0]
                )
        lo, hi = map(int, a.seeds.split(":"))
        rows = []
        for seed in range(lo, hi):
            e = MarketEngine(c, controllers)
            e.reset(seed)
            while not e.terminal():
                e.step()
            rows.append({"seed": seed, **e.metrics()})
        result = {"episodes": rows}
    elif head.group == "solve":
        from ..solvers.exact_dp import ExactBeliefDP, factorization_loss

        c = compile_market(load_spec(path))
        factory = lambda arm: ExactBeliefDP(
            c, a.focal, arm, max_nodes=a.max_nodes, node_key="belief"
        )
        if operation == "exact-dp":
            result = jsonable(factory(a.arm).solve())
        elif operation == "fact-loss":
            result = factorization_loss(factory(a.arm))
        elif operation == "info-value":
            result = {arm: jsonable(factory(arm).solve()) for arm in a.arms}
        else:
            q.error("unknown solve operation")
    elif head.group == "experiment":
        from ..eval import ExperimentRunner

        runner = ExperimentRunner(path, phase=a.phase)
        if a.workers:
            runner.config["workers"] = a.workers
        if operation == "run":
            runner.run(out)
            print(out.resolve())
            return
        from ..eval.frontier import frontier

        result = frontier(runner.config, list(map(float, a.omegas.split(","))))
    elif head.group == "pipeline":
        from ..pipeline.catalog import load_scope

        if operation == "verify-headers":
            from ..pipeline.verify import verify_headers

            if not a.source:
                q.error("verify-headers requires --source")
            result = verify_headers(
                path, a.source, count_rows=a.count_rows, inventory=a.inventory
            )
        elif operation == "build":
            from ..pipeline.compiler import compile_scope

            scope = load_scope(path, a.catalog)
            spec = compile_scope(scope)
            dump_spec(spec, out / "spec.json")
            result = {"spec_hash": compile_market(spec).spec_hash}
        elif operation == "fit-observation":
            from ..pipeline.sources.flightprices import load_panel
            from ..pipeline.observation_fit import fit_observation

            scope = load_scope(path, a.catalog)
            result = jsonable(fit_observation(load_panel(scope)))
        else:
            q.error("unknown pipeline operation")
    elif head.group == "train":
        from ..learn.ppo import train_ppo

        cfg = json.loads(Path(path).read_text())
        cfg["out"] = str(out)
        result = jsonable(train_ppo(cfg))
    elif head.group == "calibrate":
        from ..pipeline.calibrate import calibrate_scope

        result = calibrate_scope(a.scope, a.catalog, out, a.ensemble)
    else:
        from ..game.cli import run_game

        result = run_game(operation, json.loads(Path(path).read_text()))
    result_path = (
        out / (spec_output.stem + ".metadata.json")
        if spec_output
        else out / "results.json"
    )
    result_path.write_text(
        json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n"
    )
    append_manifest(
        str(out if spec_output else out.parent),
        {"kind": head.group, "operation": operation, "output": str(out.resolve())},
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
