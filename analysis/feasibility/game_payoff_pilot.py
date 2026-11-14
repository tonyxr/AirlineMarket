"""Measure cold payoff-entry costs on fixed versus declared-rule profiles (F7).

This is a payoff-cost pilot, not PSRO or a game-wide deviation certificate.
The asymmetric I0/I3 references have no B coefficient controls; that limitation
is recorded before any later game experiment chooses its policy class.
"""

from pathlib import Path
import json
import time

import numpy as np

from airline_market.campaign.artifacts import atomic_json, environment, file_hash, utc_now
from airline_market.campaign.instances import build_instance
from airline_market.core.hashing import config_hash, jsonable
from airline_market.game.payoffs import PolicyRef, controller, estimate_payoffs
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.sparse_access import sparse_access
from airline_market.spec import compile_market


def main():
    """Keep per-entry source/config identity, raw outcomes and two-cohort API checks."""
    path = Path("config/campaigns/F7_payoff_cost_pilot.json")
    cfg = json.loads(path.read_text())
    start, count = cfg["validation_seed_start"], cfg["n_cohorts"]
    if not 200000 <= start < start + count <= 300000:
        raise ValueError("game cost pilot requires validation cohorts")
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": file_hash(path), "driver_sha256": file_hash(__file__),
                "package_sources": environment()["source_hashes"]}
    atomic_json(out / "identity.json", identity)
    rows = []
    with sparse_access():
        for instance, generator, T in [("I0", "toy_hub", 10), ("I3", "hub_feeder_small", 4)]:
            spec = build_instance(generator, {"T": T, "actions": 18})
            compiled = compile_market(spec)
            pricing = next(p for p in spec.pricing if p.carrier == "A")
            rival = next(c for c in spec.controllers if c.carrier == "B")
            rival_params = {k: v for k, v in jsonable(rival).items() if k != "carrier"}
            for action_index in cfg["action_indices"]:
                ref = {"A": PolicyRef("fixed", pricing.theta_grid[action_index]),
                       "B": PolicyRef("controller", rival_params)}
                fingerprint = config_hash((identity, compiled.spec_hash, ref, start, count))
                target = out / f"{instance}_action{action_index}.json"
                if target.exists():
                    old = json.loads(target.read_text())
                    if old["fingerprint"] != fingerprint:
                        raise ValueError("changed game-cost source/config needs a new output")
                    rows.append(old)
                    continue
                values = {code: [] for code in ref}
                revenue = {code: [] for code in ref}
                started = time.perf_counter()
                cpu = time.process_time()
                for k, seed in enumerate(range(start, start + count)):
                    engine = MarketEngine(compiled, {code: controller(r, code, seed) for code, r in ref.items()})
                    engine.reset(seed)
                    while not engine.terminal():
                        engine.step()
                    for code in ref:
                        metrics = carrier_metrics(engine.world, compiled, compiled.carrier_index[code])
                        values[code].append(float(metrics["objective"]))
                        revenue[code].append(float(metrics["revenue"]))
                    atomic_json(out / "progress.json", {"updated_at": utc_now(), "instance": instance,
                                                          "action_index": action_index, "cohorts": k + 1,
                                                          "cohorts_total": count})
                seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
                check = estimate_payoffs(spec, ref, range(start, start + 2), cache={})
                errors = {code: float(np.max(np.abs(np.asarray(check.per_seed[code]) - values[code][:2])))
                          for code in ref}
                if max(errors.values()) > 1e-12:
                    raise ValueError("game-cost payoff collection disagrees with native API")
                estimates = {}
                for code in ref:
                    scale = next(o.profit_scale for o in spec.objectives if o.carrier == code)
                    payoffs_dollars = np.asarray(values[code]) * scale
                    sd = float(payoffs_dollars.std(ddof=1))
                    target_se = 0.002 * float(np.mean(revenue[code]))
                    estimates[code] = {"profit_sd_dollars": sd, "mean_revenue": float(np.mean(revenue[code])),
                                       "observed_profit_se": sd / np.sqrt(count),
                                       "target_se": target_se,
                                       "projected_n_for_target": int(np.ceil((sd / target_se) ** 2)) if target_se > 0 else None}
                row = {"status": "measured_pilot", "fingerprint": fingerprint, "instance": instance,
                       "action_index": action_index, "spec_hash": compiled.spec_hash,
                       "profile": jsonable(ref), "cohort_seeds": list(range(start, start + count)),
                       "objectives": values, "revenue": revenue, "wall_seconds": seconds,
                       "cpu_seconds": cpu_seconds, "wall_seconds_per_cohort": seconds / count,
                       "native_api_first_two_max_errors": errors, "se_sizing_projection": estimates,
                       "coefficient_dimensions": {p.carrier: len(p.theta_bounds) for p in spec.pricing},
                       "scope": "Cold payoff entry on focal fixed policies against the original threshold rival. Not a two-sided controllable-pricing game, PSRO, measured full-size n, or a Nash/deviation certificate."}
                atomic_json(target, row)
                rows.append(row)
    result = {"updated_at": utc_now(), "status": "payoff_cost_pilot_complete", "entries": len(rows),
              "n_cohorts_per_entry": count, "identity": identity,
              "entry_seconds_per_cohort": [r["wall_seconds_per_cohort"] for r in rows],
              "final_payoff_budget": None, "PSRO_measured": False,
              "remaining": "Register nondegenerate two-sided control classes; measure requested DP/MPC best-response PSRO iterations and actual target-SE payoff counts. Current I0/I3 B coefficient dimension is zero; do not silently certify that trivial response class."}
    atomic_json(out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "identity"}, indent=2))


if __name__ == "__main__":
    main()
