import copy
from .runner import ExperimentRunner


def frontier(config, omegas):
    """
    Compare objective weights and identify nondominated profit/share outcomes (Eqs.
    20–22, 29).
    """
    rows = []
    base = ExperimentRunner(config).config
    for omega in omegas:
        cfg = copy.deepcopy(base)
        cfg.setdefault("overrides", {})["objectives.0.share_weight"] = omega
        result = ExperimentRunner(cfg).run()
        for method, stats in result["summary"].items():
            rows.append(
                {
                    "omega": omega,
                    "method": method,
                    "profit": stats["profit"]["mean"],
                    "share": stats["share"]["mean"],
                }
            )
    for r in rows:
        r["non_dominated"] = not any(
            x["profit"] >= r["profit"]
            and x["share"] >= r["share"]
            and (x["profit"] > r["profit"] or x["share"] > r["share"])
            for x in rows
        )
    return rows
