"""
Flat World Model — Baseline MLP for Experiment 3 comparison.

A single MLP that takes the full state+action and predicts the full
next state.  No causal structure — used to test whether encoding
domain knowledge improves generalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlatWorldModel(nn.Module):
    """Flat (unstructured) world model — single MLP from (state, action) → next_state.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector.
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    output_dim : int
        Number of predicted variables (5 key variables by default).
    """

    def __init__(
        self,
        state_dim: int = 22,
        hidden_dim: int = 128,
        n_layers: int = 3,
        output_dim: int = 5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Input: full state vector (22) + action (1) = 23
        layers: list[nn.Module] = []
        dim = state_dim + 1
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given current state and action.

        Parameters
        ----------
        state : torch.Tensor
            Current state, shape ``(batch, 22)``.
        action : torch.Tensor
            Action integer (0–6), shape ``(batch,)`` or ``(batch, 1)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted means and log-variances for 5 key variables.
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        action_normalized = (action.float() - 3.0) / 3.0

        x = torch.cat([state, action_normalized], dim=-1)
        h = self.backbone(x)

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)

        return mean, logvar

    def loss(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL loss against ground-truth next state.

        Parameters
        ----------
        state : torch.Tensor
            Current state, shape ``(batch, 22)``.
        action : torch.Tensor
            Action, shape ``(batch,)``.
        next_state : torch.Tensor
            Ground truth next state, shape ``(batch, 22)``.

        Returns
        -------
        torch.Tensor
            Scalar NLL loss.
        """
        means, logvars = self.forward(state, action)

        # Target: same 5 key variables as structured model
        targets = torch.cat([
            next_state[:, 0:1],   # cpu
            next_state[:, 1:2],   # mem
            next_state[:, 3:4],   # pending
            next_state[:, 4:5],   # rate
            next_state[:, 6:7],   # latency
        ], dim=-1)

        var = torch.exp(logvars)
        nll = 0.5 * (logvars + (targets - means) ** 2 / var)
        return nll.mean()
