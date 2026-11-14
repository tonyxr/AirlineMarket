from .payoffs import estimate_payoffs
from ..eval.reports import statistics
from ..core.hashing import spec_hash


def ibr(
    spec, carriers, init_profile, br_solver, max_iters=20, tol=1e-3, seeds=range(10)
):
    """
    Iterate paired round-robin responses, recording convergence and profile cycles (Eqs.
    20–23).
    """
    profile = dict(init_profile)
    history = []
    seen = {}
    cycle = None
    for iteration in range(max_iters):
        key = spec_hash(profile)
        if key in seen:
            cycle = {"start": seen[key], "period": iteration - seen[key]}
            break
        seen[key] = iteration
        small = []
        for carrier in carriers:
            baseline = estimate_payoffs(spec, profile, seeds)
            candidate = br_solver(carrier, dict(profile))
            payoff = estimate_payoffs(spec, {**profile, carrier: candidate}, seeds)
            gain = statistics(
                [
                    a - b
                    for a, b in zip(
                        payoff.per_seed[carrier], baseline.per_seed[carrier]
                    )
                ]
            )
            upper = gain["ci95_normal"][1] if gain["ci95_normal"] else float("inf")
            small.append(upper < tol)
            if gain["mean"] > 0:
                profile[carrier] = candidate
            history.append(
                {
                    "iteration": iteration,
                    "carrier": carrier,
                    "gain": gain,
                    "payoffs": payoff.means,
                }
            )
        if all(small):
            return {
                "profile": profile,
                "history": history,
                "converged": True,
                "cycle": None,
            }
    return {"profile": profile, "history": history, "converged": False, "cycle": cycle}
