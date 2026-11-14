"""Stable integer indices and read-only incidence matrices (Eqs. 1–3)."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from ..core.hashing import spec_hash
from ..core.snapshot import FrozenMap
from .model import MarketSpec


@dataclass(frozen=True)
class CompiledMarket:
    """
    Validated immutable model with stable indices and read-only resource incidence (Eqs.
    1–19).
    """

    spec: MarketSpec
    carrier_index: Any
    leg_index: Any
    product_index: Any
    market_index: Any
    segment_index: Any
    airport_index: Any
    product_carrier: np.ndarray
    product_market: np.ndarray
    leg_carrier: np.ndarray
    legs_of: tuple
    products_of: tuple
    A: tuple
    A_global: np.ndarray
    market_products: tuple
    in_segment_market: np.ndarray
    segment_products: tuple
    product_cost: np.ndarray
    priority_order: tuple
    spec_hash: str


def compile_market(spec: MarketSpec) -> CompiledMarket:
    """
    Validate the network and construct stable read-only indices and incidence (Eqs.
    1–19).
    """
    from .validate import validate_spec

    validate_spec(spec)

    def indices(items, attr="id"):
        """Build a stable name-to-index map from the specification order (Eqs. 1–19)."""
        return FrozenMap({getattr(x, attr): i for i, x in enumerate(items)})

    ci, li, pi, mi, si, ai = (
        indices(spec.carriers, "code"),
        indices(spec.legs),
        indices(spec.products),
        indices(spec.markets),
        indices(spec.segments),
        indices(spec.airports, "code"),
    )
    pc = np.array([ci[p.carrier] for p in spec.products], dtype=np.int64)
    pm = np.array([mi[p.market] for p in spec.products], dtype=np.int64)
    lc = np.array([ci[l.carrier] for l in spec.legs], dtype=np.int64)
    global_a = np.zeros((len(li), len(pi)), dtype=np.int8)
    for j, p in enumerate(spec.products):
        for leg in p.legs:
            global_a[li[leg], j] = 1
    legs = tuple(tuple(np.flatnonzero(lc == i).tolist()) for i in range(len(ci)))
    prods = tuple(tuple(np.flatnonzero(pc == i).tolist()) for i in range(len(ci)))
    matrices = tuple(global_a[np.ix_(legs[i], prods[i])].copy() for i in range(len(ci)))
    market_prods = tuple(
        tuple(np.flatnonzero(pm == m).tolist()) for m in range(len(mi))
    )
    ins = np.array([m.in_segment for m in spec.markets], dtype=bool)
    cost = np.array(
        [
            p.service_cost + sum(spec.legs[li[l]].cost_per_pax for l in p.legs)
            for p in spec.products
        ]
    )
    priorities = tuple(
        tuple(sorted(prods[i], key=lambda j: (spec.products[j].priority, j)))
        for i in range(len(ci))
    )
    for array in (pc, pm, lc, global_a, ins, cost, *matrices):
        array.setflags(write=False)
    return CompiledMarket(
        spec,
        ci,
        li,
        pi,
        mi,
        si,
        ai,
        pc,
        pm,
        lc,
        legs,
        prods,
        matrices,
        global_a,
        market_prods,
        ins,
        tuple(j for j in range(len(pi)) if ins[pm[j]]),
        cost,
        priorities,
        spec_hash(spec),
    )
