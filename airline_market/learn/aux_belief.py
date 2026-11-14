"""Training-only joint-event heads and probability calibration (R3)."""

import torch
from torch import nn
import numpy as np


class JointEventHead(nn.Module):
    """
    Training-only binary joint-availability auxiliary prediction head (Eqs. 18–19,
    26–27).
    """

    def __init__(self, hidden_dim, n_events):
        """
        Bind declared inputs and initialize JointEventHead bookkeeping (Eqs. 18–19,
        26–27).
        """
        super().__init__()
        self.head = nn.Linear(hidden_dim, n_events)

    def forward(self, hidden):
        """
        Compute the declared neural outputs from the supplied feature tensors (Eqs.
        18–19, 26–27).
        """
        return self.head(hidden)

    def loss(self, hidden, labels, weight=1.0):
        """Compute the auxiliary binary-event prediction loss (Eqs. 18–19, 26–27)."""
        return weight * nn.functional.binary_cross_entropy_with_logits(
            self(hidden), labels.float()
        )


def event_labels(kernel, world, focal, events):
    """
    Build training-only joint-event targets from a realized simulator state (Eqs. 18–19,
    26–27).
    """
    from ..belief import ExactBelief

    arm = next(
        a.name
        for a in kernel.compiled.spec.information
        if a.observer == kernel.compiled.spec.carriers[focal].code
    )
    b = ExactBelief(
        kernel.compiled, kernel.compiled.spec.carriers[focal].code, arm, kernel
    )
    h = b._hidden(kernel.encode(world, focal))
    return np.array([float(e.evaluate(h, kernel.compiled, focal)) for e in events])


def calibration(probabilities, targets, bins=10):
    """
    Report Brier error and probability calibration-bin statistics (Eqs. 18–19, 26–27).
    """
    p = np.asarray(probabilities)
    y = np.asarray(targets)
    rows = []
    for k in range(bins):
        selected = (p >= k / bins) & (p < (k + 1) / bins if k < bins - 1 else p <= 1.0)
        if selected.any():
            rows.append(
                {
                    "bin": k,
                    "n": int(selected.sum()),
                    "predicted": float(p[selected].mean()),
                    "target": float(y[selected].mean()),
                }
            )
    return {"brier": float(np.mean((p - y) ** 2)), "curve": rows}
