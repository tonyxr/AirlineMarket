"""Separate network-opportunity-cost policy class (R1)."""

import torch
from torch import nn
from ..core.money import round_fare


class StructuredActorCritic(nn.Module):
    """
    Separate network-opportunity-cost actor and additive leg-value critic (Eqs. 18–19,
    26–27).
    """

    policy_class = "network_opportunity_cost"

    def __init__(self, leg_dim, global_dim, n_coefficients, hidden=32):
        """
        Bind declared inputs and initialize StructuredActorCritic bookkeeping (Eqs.
        18–19, 26–27).
        """
        super().__init__()
        self.leg_encoder = nn.Sequential(
            nn.Linear(leg_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.dual_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(global_dim, n_coefficients)
        self.leg_value = nn.Linear(hidden, 1)
        self.global_value = nn.Linear(global_dim, 1)

    def forward(self, legs, global_features):
        """
        Compute the declared neural outputs from the supplied feature tensors (Eqs.
        18–19, 26–27).
        """
        h = self.leg_encoder(legs)
        pi = torch.nn.functional.softplus(self.dual_head(h).squeeze(-1))
        return (
            pi,
            self.residual_head(global_features),
            self.global_value(global_features).squeeze(-1)
            + self.leg_value(h).sum(-2).squeeze(-1),
        )

    @staticmethod
    def price_tensor(base, incidence, pi, features, theta, low, high):
        """
        Map differentiable leg values and residual coefficients through incidence (Eqs.
        18–19, 26–27).
        """
        return torch.clamp(
            base + pi @ incidence + torch.einsum("...pk,...k->...p", features, theta),
            min=low,
            max=high,
        )

    @staticmethod
    def executable_prices(tensor):
        """
        Convert bounded model prices to the shared cent currency convention (Eqs. 18–19,
        26–27).
        """
        return [round_fare(x) for x in tensor.detach().cpu().tolist()]


def pretrain_duals(model, legs, globals, target, steps=500, lr=0.01):
    """
    Fit leg opportunity costs to a known LP-dual target for a sanity check (Eqs. 18–19,
    26–27).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pi, _, _ = model(legs, globals)
        loss = (pi - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())
