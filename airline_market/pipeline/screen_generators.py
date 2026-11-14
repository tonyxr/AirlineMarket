"""R5: opt-in v2 I4 parameters, with byte-identical legacy defaults.

price_scale multiplies all latent-class price-sensitivity coefficients alpha.
A positive outside utility shift subtracts from every inside market utility,
using the native zero-outside normalization. Explicit capacity overrides change
fixed carrier legs; capacity_support remains the declared stochastic prior.
"""

import math
from dataclasses import replace

from airline_market.campaign.instances import grid_indices
from airline_market.core.hashing import jsonable
from airline_market.pipeline.generators import hub_feeder_family as legacy_family
from airline_market.spec import load_spec

EXTENSIONS = {"price_scale": 1.0, "outside_utility_shift": 0.0}


def hub_feeder_family(params=None):
    """Extend I4 without changing captured legacy generators or default hashes."""
    p = dict(params or {})
    capacity = p.get("capacity", 3)
    scale = p.pop("price_scale", 1.0)
    shift = p.pop("outside_utility_shift", 0.0)
    own = p.pop("capacity_focal", capacity)
    rival = p.pop("capacity_rival", capacity)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
        raise ValueError("price_scale must be positive and outside shift finite")
    for value in (own, rival):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("carrier capacities must be nonnegative integers")
    original = legacy_family(p)
    if scale == 1 and shift == 0 and own == capacity and rival == capacity:
        return original
    d = jsonable(original)
    for leg in d["legs"]:
        value = own if leg["carrier"] == "A" else rival
        if value != capacity and len(leg["capacity"]["support"]) == 1:
            leg["capacity"]["support"] = [value]
    for segment in d["segments"]:
        for taste in segment["classes"]:
            taste["alpha"] *= scale
            taste["market_value"] = {
                key: value - shift for key, value in taste["market_value"].items()
            }
    d["metadata"]["generator"]["params"].update(
        price_scale=scale,
        outside_utility_shift=shift,
        capacity_focal=own,
        capacity_rival=rival,
    )
    return load_spec(d)


def screen_spec(parameters):
    """Resolve v2 screen parameters and the original nested ordered action grid."""
    p = dict(parameters)
    actions = p.pop("actions", 4)
    if "T" in p:
        p["periods"] = p.pop("T")
    spec = hub_feeder_family(p)
    first = spec.pricing[0]
    return replace(
        spec,
        pricing=(
            replace(
                first,
                theta_grid=tuple(first.theta_grid[i] for i in grid_indices(actions)),
            ),
            *spec.pricing[1:],
        ),
    )
