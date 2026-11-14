"""Named instance construction and nested action grids (plan Appendix B)."""

from dataclasses import replace
from pathlib import Path

from ..spec import load_spec
from ..spec.library_builders import toy_hub
from ..pipeline.generators import hub_feeder_family, independent_legs


def grid_indices(size: int) -> tuple[int, ...]:
    """Return nested subsets of the original 18-action grid, preserving ties."""
    order = [0, 17, 8, 13, 4, 11, 1, 6, 15, 2, 3, 5, 7, 9, 10, 12, 14, 16]
    if size < 1 or size > len(order):
        raise ValueError("grid size must lie in 1..18")
    return tuple(sorted(order[:size]))


def build_instance(generator: str, parameters: dict):
    """Build the declared model; reject unsupported catalog changes explicitly."""
    params = dict(parameters.get("instance_params", {}))
    params.update(
        {
            key: value
            for key, value in parameters.items()
            if key
            not in {
                "arm",
                "backend",
                "actions",
                "grid_indices",
                "n_cohorts",
                "seed_start",
                "instance_params",
            }
        }
    )
    if generator in ("toy_hub", "toy_hub_reduced"):
        from ..legacy_toy.model import Config

        T = int(params.get("T", params.get("periods", 10)))
        spec = toy_hub(Config(horizon=T, shock_step=min(5, T - 1)))
    elif generator in ("hub_feeder_small", "hub_feeder_family", "independent_legs"):
        if generator == "hub_feeder_small":
            params = {"n_spokes": 2, "periods": 4, **params}
        if "T" in params:
            params["periods"] = params.pop("T")
        prior = params.pop("rival_capacity_prior", None)
        if prior is not None:
            params["capacity_support"] = {
                "narrow": [2, 3],
                "wide": [1, 3],
                "trimodal": [1, 2, 3],
            }[prior]
        bottleneck = params.get("bottleneck")
        if bottleneck in ("focal_hub", "rival_feeder"):
            params["bottleneck"] = {"focal_hub": "focal", "rival_feeder": "rival"}[
                bottleneck
            ]
        spec = (
            independent_legs if generator == "independent_legs" else hub_feeder_family
        )(params)
    elif Path(generator).is_file():
        spec = load_spec(generator)
    else:
        raise ValueError(f"instance generator not implemented: {generator}")
    size = parameters.get("actions")
    indices = parameters.get("grid_indices")
    if indices is not None or size is not None:
        indices = tuple(indices) if indices is not None else grid_indices(int(size))
        pricing = spec.pricing[0]
        spec = replace(
            spec,
            pricing=(
                replace(
                    pricing, theta_grid=tuple(pricing.theta_grid[k] for k in indices)
                ),
                *spec.pricing[1:],
            ),
        )
    return spec
