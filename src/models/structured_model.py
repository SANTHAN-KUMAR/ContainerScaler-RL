"""
Structured World Model — Causal sub-network architecture for Experiment 3.

5 sub-networks, each predicting one variable from its causal parents:
  predict_cpu(current_cpu, request_rate, new_replicas)      → next_cpu
  predict_mem(current_mem, request_rate, new_replicas)      → next_mem
  predict_pending(current_pending, delta)                   → next_pending
  predict_request_rate(current_rate, derivative, time)      → next_rate
  predict_latency(pred_cpu, pred_mem, pred_pending, queue)  → next_latency

Each sub-network is a GaussianMLP outputting (mean, log_variance).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianMLP(nn.Module):
    """Small MLP that outputs a Gaussian distribution (mean + log-variance).

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Width of hidden layers.
    output_dim : int
        Dimensionality of the predicted variable.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_variance) predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance, each of shape ``(batch, output_dim)``.
        """
        h = self.backbone(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # Clamp log-variance for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        return mean, logvar

    def nll_loss(
        self, x: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for Gaussian output.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        target : torch.Tensor
            Ground-truth target values.

        Returns
        -------
        torch.Tensor
            Scalar NLL loss.
        """
        mean, logvar = self.forward(x)
        var = torch.exp(logvar)
        # NLL = 0.5 * (log(var) + (target - mean)^2 / var)
        nll = 0.5 * (logvar + (target - mean) ** 2 / var)
        return nll.mean()


class StructuredWorldModel(nn.Module):
    """Causal-structure world model with 5 specialized sub-networks.

    Encodes domain knowledge: replicas → capacity → utilization → latency.
    Each sub-network predicts from its causal parents only.

    Parameters
    ----------
    hidden_dim : int
        Width of hidden layers in each sub-network.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()

        # Sub-network 1: CPU utilization
        # Inputs: current_cpu, request_rate, new_replicas → next_cpu
        self.cpu_net = GaussianMLP(input_dim=3, hidden_dim=hidden_dim)

        # Sub-network 2: Memory utilization
        # Inputs: current_mem, request_rate, new_replicas → next_mem
        self.mem_net = GaussianMLP(input_dim=3, hidden_dim=hidden_dim)

        # Sub-network 3: Pending pods
        # Inputs: current_pending, delta → next_pending
        self.pending_net = GaussianMLP(input_dim=2, hidden_dim=hidden_dim)

        # Sub-network 4: Request rate
        # Inputs: current_rate, derivative, time → next_rate
        self.rate_net = GaussianMLP(input_dim=3, hidden_dim=hidden_dim)

        # Sub-network 5: Latency
        # Inputs: pred_cpu, pred_mem, pred_pending, queue_depth → next_latency
        self.latency_net = GaussianMLP(input_dim=4, hidden_dim=hidden_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given current state and action.

        Parameters
        ----------
        state : torch.Tensor
            Current 22-dim observation vector, shape ``(batch, 22)``.
        action : torch.Tensor
            Action integer (0–6), shape ``(batch, 1)`` or ``(batch,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Predicted next state means and variances for the 5 key variables:
            (cpu, mem, pending, rate, latency) — shape ``(batch, 5)`` each.
        """
        # Extract relevant features from state vector
        cpu = state[:, 0:1]           # cpu_util
        mem = state[:, 1:2]           # mem_util
        replicas = state[:, 2:3]      # replicas / max
        pending = state[:, 3:4]       # pending / 10
        rate = state[:, 4:5]          # request_rate / 500
        deriv = state[:, 5:6]         # traffic derivative / 100
        queue = state[:, 8:9]         # queue_depth / 10000

        # Time encoding
        time_sin = state[:, 18:19]
        state[:, 19:20]
        time = time_sin  # use sin component as time proxy

        # Action delta (normalized)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        delta = (action.float() - 3.0) / 3.0  # normalize to [-1, 1]

        # New replicas after action
        new_replicas = replicas + delta / 30.0  # approximate

        # ── Chain sub-networks ───────────────────────────────────────
        # 1. CPU
        cpu_input = torch.cat([cpu, rate, new_replicas], dim=-1)
        cpu_mean, cpu_logvar = self.cpu_net(cpu_input)

        # 2. Memory
        mem_input = torch.cat([mem, rate, new_replicas], dim=-1)
        mem_mean, mem_logvar = self.mem_net(mem_input)

        # 3. Pending pods
        pending_input = torch.cat([pending, delta], dim=-1)
        pending_mean, pending_logvar = self.pending_net(pending_input)

        # 4. Request rate
        rate_input = torch.cat([rate, deriv, time], dim=-1)
        rate_mean, rate_logvar = self.rate_net(rate_input)

        # 5. Latency (uses predictions from above)
        latency_input = torch.cat([cpu_mean, mem_mean, pending_mean, queue], dim=-1)
        latency_mean, latency_logvar = self.latency_net(latency_input)

        # Assemble predictions
        means = torch.cat([cpu_mean, mem_mean, pending_mean, rate_mean, latency_mean], dim=-1)
        logvars = torch.cat(
            [cpu_logvar, mem_logvar, pending_logvar, rate_logvar, latency_logvar], dim=-1,
        )

        return means, logvars

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

        # Extract target variables from next_state
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
