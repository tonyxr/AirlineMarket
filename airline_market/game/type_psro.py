"""Population expansion with genuine episode-mixture DP best responses."""

import time

import numpy as np

from .meta_solvers import solve_bimatrix
from .type_belief import EpisodeTypeDP, HistoryController


def exact_type_psro(
    compiled, populations, *, max_iters=12, tol=1e-6, max_nodes=100_000, progress=None
):
    """Audit deviations over all finite own-history policies on both action grids.

    The matrix is only the population meta-game. The stopping test is a new
    joint episode-type DP best response for each carrier against the opponent's
    complete meta-mixture. No matrix-only deviation certificate is used.
    """
    carriers = tuple(c.code for c in compiled.spec.carriers)
    if len(carriers) != 2 or set(populations) != set(carriers):
        raise ValueError("two complete carrier populations required")
    if any(
        len(p.theta_grid) < 2 or not p.controllable_features
        for p in compiled.spec.pricing
    ):
        raise ValueError("two-sided PSRO requires nondegenerate pricing controls")
    populations = {c: list(populations[c]) for c in carriers}
    for c in carriers:
        grid = next(p.theta_grid for p in compiled.spec.pricing if p.carrier == c)
        if (
            not populations[c]
            or any(p.grid != grid for p in populations[c])
            or len({p.arm for p in populations[c]}) != 1
        ):
            raise ValueError("population policy must use its carrier's declared grid")
    cache = {}
    history = []
    started = time.perf_counter()
    for iteration in range(max_iters):
        a, b = carriers
        A = np.zeros((len(populations[a]), len(populations[b])))
        B = np.zeros_like(A)
        for i, left in enumerate(populations[a]):
            for j, right in enumerate(populations[b]):
                key = left.hash(), right.hash()
                if key not in cache:
                    values = []
                    for focal, own, rival, other in [
                        (a, left, b, right),
                        (b, right, a, left),
                    ]:
                        dp = EpisodeTypeDP(
                            compiled,
                            focal,
                            own.arm,
                            [{rival: HistoryController(other)}],
                            [1],
                            max_nodes=max_nodes,
                        )
                        values.append(dp.evaluate(own))
                    cache[key] = values
                    if progress:
                        progress(
                            {
                                "stage": "payoff",
                                "iteration": iteration,
                                "entries": len(cache),
                                "seconds": time.perf_counter() - started,
                            }
                        )
                A[i, j], B[i, j] = cache[key]
        meta = solve_bimatrix(A, B)
        responses, values, nodes, times = {}, [], [], []
        for k, focal in enumerate(carriers):
            rival = carriers[1 - k]
            probabilities = meta["strategies"][1 - k]
            active = [
                (p, float(w))
                for p, w in zip(populations[rival], probabilities)
                if w > 0
            ]
            begin = time.perf_counter()
            dp = EpisodeTypeDP(
                compiled,
                focal,
                populations[focal][0].arm,
                [{rival: HistoryController(p)} for p, _ in active],
                [w for _, w in active],
                max_nodes=max_nodes,
            )
            values.append(dp.solve())
            responses[focal] = dp.policy()
            nodes.append(len(dp.nodes))
            times.append(time.perf_counter() - begin)
            if progress:
                progress(
                    {
                        "stage": "best_response",
                        "iteration": iteration,
                        "focal": focal,
                        "nodes": nodes[-1],
                        "seconds": times[-1],
                    }
                )
        gains = [v - m for v, m in zip(values, meta["payoffs"])]
        row = {
            "iteration": iteration,
            "population_sizes": [len(populations[c]) for c in carriers],
            "mixtures": [x.tolist() for x in meta["strategies"]],
            "payoffs": list(meta["payoffs"]),
            "best_response_values": values,
            "deviation_gains": gains,
            "best_response_nodes": nodes,
            "best_response_seconds": times,
            "payoff_entries": len(cache),
        }
        history.append(row)
        if progress:
            progress({"stage": "iteration", **row})
        if min(gains) < -1e-9:
            raise ValueError("best response falls below an available population policy")
        if max(gains) <= tol:
            return {
                "status": "converged",
                "history": history,
                "certified": True,
                "scope": "All deterministic permitted-own-history policies on the registered finite coefficient grids; independent episode mixtures; declared physical model only.",
                "max_deviation_gain": max(gains),
                "payoff_entries": len(cache),
                "wall_seconds": time.perf_counter() - started,
                "populations": populations,
            }
        added = False
        for k, focal in enumerate(carriers):
            response = responses[focal]
            if gains[k] > tol and response.hash() not in {
                p.hash() for p in populations[focal]
            }:
                populations[focal].append(response)
                added = True
        if not added:
            raise ValueError(
                "positive deviation but response population did not expand"
            )
    return {
        "status": "iteration_limit",
        "history": history,
        "certified": False,
        "max_deviation_gain": max(gains),
        "payoff_entries": len(cache),
        "wall_seconds": time.perf_counter() - started,
        "populations": populations,
    }
