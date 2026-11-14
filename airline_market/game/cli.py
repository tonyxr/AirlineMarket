from ..spec import load_spec
from ..core.hashing import jsonable
from .payoffs import PolicyRef, estimate_payoffs
from .ibr import ibr
from .psro import psro


def run_game(operation, config):
    """Run the requested finite-policy competition experiment (Eqs. 20–23)."""
    spec = load_spec(config["spec"])
    seeds = tuple(range(*config.get("seeds", [0, 20])))
    carriers = [c.code for c in spec.carriers]
    grids = {
        i: [
            PolicyRef("fixed", x)
            for x in next(p.theta_grid for p in spec.pricing if p.carrier == i)
        ]
        if next(p.theta_grid for p in spec.pricing if p.carrier == i)
        else [
            PolicyRef(
                "controller",
                {
                    "controller": next(
                        a.controller for a in spec.controllers if a.carrier == i
                    ),
                    "params": dict(
                        next(a.params for a in spec.controllers if a.carrier == i)
                    ),
                },
            )
        ]
        for i in carriers
    }

    def best_response(i, profile):
        """
        Choose or construct a response to the frozen opponent profile (Eqs. 20–23).
        """
        return max(
            grids[i],
            key=lambda p: estimate_payoffs(spec, {**profile, i: p}, seeds).means[i],
        )

    if operation == "psro":
        return jsonable(
            psro(
                spec,
                grids,
                seeds,
                br_solver=best_response,
                max_iters=config.get("max_iters", 10),
            )
        )
    if operation == "ibr":
        return jsonable(
            ibr(
                spec,
                carriers,
                {i: g[0] for i, g in grids.items()},
                best_response,
                max_iters=config.get("max_iters", 20),
                seeds=seeds,
            )
        )
    raise ValueError("unknown game operation")
