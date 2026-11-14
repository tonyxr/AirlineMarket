"""S1: reach-weighted pairwise information and common-prefix policy divergence.

Only saved, reachable optimal-policy actions are read. The native finite model
enumerates their transitions; Bellman optimization, sampling and actor calls
are never used. Information is in nats, separately for every rival-leg pair.
"""

import gzip
import math
import pickle
from collections import defaultdict
from functools import lru_cache
from itertools import combinations

from airline_market.solvers.finite_dp import FiniteBeliefSolver


def read_policy(path, expected_records):
    """Read a trusted local producer's pickle stream; reject duplicate records."""
    records = {}
    with gzip.open(path, "rb") as stream:
        while True:
            try:
                node, index = pickle.load(stream)
            except EOFError:
                break
            if node in records:
                raise ValueError("duplicate saved policy node")
            records[node] = index
    if len(records) != expected_records:
        raise ValueError("incomplete saved policy stream")
    return records


def analyze_settled_setting(root, setting_id):
    """Read only returned publisher futures; retain hashes of saved policy bytes.

    The policy stream was not transitively sealed by the original producer.
    Its hash here is explicitly a post-completion snapshot, not a retroactive
    original seal. No policy file is regenerated or original result modified.
    Callers reserve memory for this exact traversal as for a screen cell.
    """
    import json
    from pathlib import Path

    from airline_market.campaign.artifacts import file_hash
    from airline_market.campaign.settled_runner import completed_snapshot
    from airline_market.solvers.lagged_dp import history_dp
    from airline_market.spec import compile_market, load_spec

    root = Path(root)
    acknowledged = set(json.loads((root / "settled_cells.json").read_bytes())["cell_ids"])
    cells = [c for c in json.loads((root / "cells.json").read_bytes()) if c["parameters"]["setting_id"] == setting_id]
    if not cells or any(c["cell_id"] not in acknowledged for c in cells):
        raise ValueError("setting is not fully acknowledged by returned publisher futures")
    if {c["parameters"]["arm"] for c in cells} != {"direct", "linked", "unrelated", "oracle", "full_state"} or len(cells) != 5:
        raise ValueError("setting does not contain exactly the five registered arms")
    source_hashes = {c["parameters"]["source_spec_sha256"] for c in cells}
    if len(source_hashes) != 1:
        raise ValueError("arm physical sources differ")
    dps, policies, provenance, information = {}, {}, {}, {}
    for cell in cells:
        found = completed_snapshot(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found is None:
            raise ValueError("acknowledged setting is missing its terminal receipt")
        directory, result = found
        if result["status"] != "solved":
            raise ValueError("policy metrics require solved arms; retain limited values as null")
        parameters = cell["parameters"]
        source = Path(parameters["source_spec"])
        if file_hash(source) != parameters["source_spec_sha256"]:
            raise ValueError("physical source changed")
        compiled = compile_market(load_spec(source))
        if compiled.spec_hash != result["spec_hash"]:
            raise ValueError("saved compiled physical identity differs")
        arm = parameters["arm"]
        dp = history_dp(compiled, "A", arm, max_nodes=0)
        path = directory / "optimal_policy.pkl.gz"
        policy_hash = file_hash(path)
        records = read_policy(path, result["policy_records"])
        if file_hash(path) != policy_hash:
            raise ValueError("policy changed while reading")
        dps[arm], policies[arm] = dp, records
        information[arm] = policy_information(dp, records)
        provenance[arm] = {
            "cell_id": cell["cell_id"],
            "config_hash": cell["config_hash"],
            "source_sha256": parameters["source_spec_sha256"],
            "receipt_sha256": file_hash(directory / "COMPLETE.json"),
            "results_sha256": file_hash(directory / "results.json"),
            "policy_post_completion_sha256": policy_hash,
        }
    return {
        "plan_id": "S1",
        "setting_id": setting_id,
        "provenance": provenance,
        "policy_integrity_scope": "Policy hashes are post-completion snapshots, not original transitive seals",
        "pairwise_information": information,
        "policy_divergence": first_price_divergence(dps["direct"], dps["linked"], policies["direct"], policies["linked"]),
        "scientific_budget": None,
    }


def pairwise_information(belief):
    """Marginalize memory/journeys, then compute each stock pair's raw MI.

    The supplied posterior must already have unit mass. No mass repair,
    clipping of small signed roundoff, or sum over all pairs is substituted
    for a particular pair's mutual information.
    """
    mass = math.fsum(weight for _, weight in belief)
    if not belief or any(w < 0 or not math.isfinite(w) for _, w in belief):
        raise ValueError("invalid posterior weights")
    if abs(mass - 1) > 1e-12:
        raise ValueError("posterior mass is not one")
    stocks = defaultdict(list)
    for hidden, weight in belief:
        stocks[hidden[0]].append(weight)
    law = {stock: math.fsum(weights) for stock, weights in stocks.items()}
    width = len(next(iter(law)))
    if any(len(stock) != width for stock in law):
        raise ValueError("inconsistent rival stock coordinates")
    result = {}
    for a, b in combinations(range(width), 2):
        joint, left, right = defaultdict(list), defaultdict(list), defaultdict(list)
        for stock, weight in law.items():
            joint[stock[a], stock[b]].append(weight)
            left[stock[a]].append(weight)
            right[stock[b]].append(weight)
        joint = {k: math.fsum(v) for k, v in joint.items()}
        left = {k: math.fsum(v) for k, v in left.items()}
        right = {k: math.fsum(v) for k, v in right.items()}
        result[a, b] = math.fsum(
            p * (math.log(p) - math.log(left[x]) - math.log(right[y]))
            for (x, y), p in joint.items()
            if p > 0
        )
    return result


def projected_roots(dp):
    """Retain the original initial-observation law, without solving a value."""
    return tuple(
        (p, dp.finite.model.project(node.observation, node.belief))
        for p, node in dp.roots()
    )


def _index(records, node, grid):
    index = records[node]
    if type(index) is not int or not 0 <= index < len(grid):
        raise ValueError("invalid saved policy action")
    return index


def _check_mass(mass):
    if not math.isfinite(mass) or abs(mass - 1) > 1e-12:
        raise ValueError("policy reach mass is not one")


def policy_information(dp, records):
    """Exact unconditional reach-weighted MI, including the terminal posterior.

    No divide by reach mass is performed. Each period should already have
    total probability one, and that raw mass is reported and checked.
    """
    model = dp.finite.model
    solver = FiniteBeliefSolver(model, dp.grid, max_nodes=0)
    current = defaultdict(float)
    for probability, node in projected_roots(dp):
        current[node] += probability
    labels = [dp.compiled.spec.legs[j].id for j in dp.compiled.legs_of[model.rival]]
    pairs = list(combinations(range(len(labels)), 2))
    periods = []
    visited = set()
    for t in range(model.T + 1):
        mass = math.fsum(current.values())
        _check_mass(mass)
        terms = {pair: [] for pair in pairs}
        following = defaultdict(float)
        for node, reach in current.items():
            if node[0] != t or reach < 0:
                raise ValueError("invalid saved policy clock or reach")
            for pair, mi in pairwise_information(node[-1]).items():
                terms[pair].append(reach * mi)
            if t < model.T:
                index = _index(records, node, dp.grid)
                visited.add(node)
                for probability, _, child in solver.successors(node, index):
                    following[child] += reach * probability
        periods.append(
            {
                "period": t,
                "terminal": t == model.T,
                "reach_mass": mass,
                "states": len(current),
                "pairwise_mi_nats": [math.fsum(terms[pair]) for pair in pairs],
            }
        )
        current = following
    if visited != records.keys():
        raise ValueError("saved policy includes unreachable or terminal records")
    return {
        "plan_id": "S1",
        "definition": "Unconditional reach-weighted pairwise rival-stock MI; nats; no mass renormalization",
        "pairs": [[labels[a], labels[b]] for a, b in pairs],
        "periods": periods,
        "policy_records": len(records),
        "bellman_or_sampling_calls": 0,
    }


def first_price_divergence(direct, linked, direct_records, linked_records):
    """Couple physical paths until their first posted-fare-vector difference.

    Every action reads only that arm's saved observation/posterior node.
    Hidden state is used solely for analyst-side coupling and observations.
    Equality means the exact native cent-rounded vector, including all focal
    products. A coefficient alias is reported separately. Once fares differ,
    that branch stops: histories after different actions are never compared.
    """
    dm, lm = direct.finite.model, linked.finite.model
    if (
        direct.compiled.spec_hash != linked.compiled.spec_hash
        or dm.arm.name != "direct"
        or lm.arm.name != "linked"
        or direct.grid != linked.grid
        or dm.focal != lm.focal
        or not dm.history_targets <= lm.history_targets
        or not {q.id for q in dm.requests} <= {q.id for q in lm.requests}
    ):
        raise ValueError("requires nested direct/linked arms of the same physical source/grid")
    if len(set(dm.own_uses)) != len(dm.own_uses):
        raise ValueError("coupling requires distinct own-product leg incidences")
    ds = FiniteBeliefSolver(dm, direct.grid, max_nodes=0)
    ls = FiniteBeliefSolver(lm, linked.grid, max_nodes=0)

    def project_hidden(hidden):
        return (*hidden[:2], dm._remember(hidden[2])) if dm.has_lag else hidden[:2]

    def signature(node):
        return node[:6]

    droots = {signature(n): (p, n) for p, n in projected_roots(direct)}
    initial = defaultdict(float)
    direct_masses = defaultdict(float)
    for probability, ln in projected_roots(linked):
        for hidden, weight in ln[-1]:
            signal = dm.signal(0, project_hidden(hidden), ln[4])
            _, dn = droots[(*ln[:5], signal)]
            initial[dn, ln, hidden, False] += probability * weight
            direct_masses[dn] += probability * weight
    for probability, node in droots.values():
        if abs(direct_masses[node] - probability) > 1e-12:
            raise ValueError("initial coupling does not reproduce direct observations")

    @lru_cache(None)
    def children(which, node, index):
        solver = ds if which == "direct" else ls
        result = {}
        for _, _, child in solver.successors(node, index):
            delta = tuple(a - b for a, b in zip(node[2], child[2]))
            sale = -1 if not any(delta) else dm.own_uses.index(delta)
            key = child[2], child[3], child[5], sale
            if key in result:
                raise ValueError("ambiguous native observation child")
            result[key] = child
        return result

    current = initial
    price_first = [0.0] * lm.T
    coefficient_first = [0.0] * lm.T
    prefix_mass = []
    for t in range(lm.T):
        prefix_mass.append(math.fsum(current.values()))
        following = defaultdict(float)
        price_terms, coefficient_terms = [], []
        for (dn, ln, hidden, seen_coefficient), reach in current.items():
            if dn[0] != t or ln[0] != t:
                raise ValueError("coupled policy clocks differ")
            di = _index(direct_records, dn, direct.grid)
            li = _index(linked_records, ln, linked.grid)
            dp = dm.prices(direct.grid[di], t, dn[1], dn[2], dn[5])
            lp = lm.prices(linked.grid[li], t, ln[1], ln[2], ln[5])
            if di != li and not seen_coefficient:
                coefficient_terms.append(reach)
                seen_coefficient = True
            if dp != lp:
                price_terms.append(reach)
                continue
            dc, lc = children("direct", dn, di), children("linked", ln, li)
            for p, own, journeys, new_hidden, _, sale in lm.step(
                t, ln[2], ln[3], hidden, lp, ln[4]
            ):
                if p <= 0:
                    raise ValueError("native step yielded nonpositive probability")
                dsig = dm.signal(t + 1, project_hidden(new_hidden), dp)
                lsig = lm.signal(t + 1, new_hidden, lp)
                dchild = dc[own, journeys, dsig, sale]
                lchild = lc[own, journeys, lsig, sale]
                if new_hidden not in dict(lchild[-1]) or project_hidden(new_hidden) not in dict(dchild[-1]):
                    raise ValueError("coupled truth is outside the native posterior support")
                following[dchild, lchild, new_hidden, seen_coefficient] += reach * p
        price_first[t] = math.fsum(price_terms)
        coefficient_first[t] = math.fsum(coefficient_terms)
        current = following
        _check_mass(math.fsum(price_first) + math.fsum(current.values()))
    never = math.fsum(current.values())
    return {
        "plan_id": "S1",
        "definition": "First native posted-fare-vector difference on a common physical price prefix; zero-based periods",
        "first_period": next((t for t, p in enumerate(price_first) if p > 0), None),
        "first_price_difference_mass_by_period": price_first,
        "no_price_difference_mass": never,
        "total_mass": math.fsum(price_first) + never,
        "common_prefix_mass_by_period": prefix_mass,
        "coefficient_difference_first_mass_before_price_split": coefficient_first,
        "coefficient_scope": "Separate action-index diagnostic, censored after first fare difference; aliases do not count as fare differences",
        "bellman_or_sampling_calls": 0,
    }
