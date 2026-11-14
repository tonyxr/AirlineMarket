"""S1-r1: thirteen-run five-factor DSD crossed with bottleneck and lag.

The Paley order-six conference construction gives 12 mirrored noncenter
runs plus one center. Deleting its sixth column leaves five orthogonal
linear factors; linear effects are orthogonal to every quadratic/interaction.
"""

from itertools import product

import numpy as np

FACTORS = {
    "connecting_share": (0.3, 0.6, 0.9),
    "arrival_probability": (0.6, 0.8, 0.95),
    "capacity": (2, 3, 4),
    "capacity_support": ((2, 3), (1, 3), (1, 2, 3)),
    "price_scale": (0.5, 1.0, 2.0),
}


def dsd_matrix():
    """Construct the fixed conference DSD without design search or outcome selection."""
    c = np.zeros((6, 6), dtype=int)
    c[0, 1:] = 1
    c[1:, 0] = 1
    for i in range(5):
        for j in range(5):
            if i != j:
                c[i + 1, j + 1] = 1 if (i - j) % 5 in (1, 4) else -1
    return np.vstack((c[:, :5], -c[:, :5], np.zeros((1, 5), dtype=int)))


def round_one():
    """Return every prespecified setting in a stable, non-random order."""
    rows = []
    for i, coded in enumerate(dsd_matrix()):
        p = {
            key: levels[int(code) + 1]
            for (key, levels), code in zip(FACTORS.items(), coded)
        }
        for bottleneck, lag in product(("rival", "both"), (0, 1)):
            rows.append(
                {
                    "setting_id": f"r1-{i + 1:02d}-{bottleneck}-lag{lag}",
                    "dsd_row": i + 1,
                    "coded": list(map(int, coded)),
                    "parameters": {
                        **p,
                        "bottleneck": bottleneck,
                        "lag_periods": lag,
                        "n_spokes": 2,
                        "T": 5,
                        "actions": 4,
                        "rival_controller": "threshold_linear_rule",
                    },
                }
            )
    return rows
