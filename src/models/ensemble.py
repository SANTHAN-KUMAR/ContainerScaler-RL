"""
Model Ensemble — Wraps N world model instances for epistemic uncertainty.

Epistemic uncertainty is estimated via inter-model variance: when models
disagree, the prediction is uncertain.  Used in Experiment 6 for
calibration analysis.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


class ModelEnsemble(nn.Module):
    """Ensemble of N world models for uncertainty estimation.

    Parameters
    ----------
    model_class : Type[nn.Module]
        The world model class to instantiate (StructuredWorldModel or FlatWorldModel).
    n_models : int
        Number of ensemble members.
    **model_kwargs
        Keyword arguments passed to each model constructor.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        n_models: int = 5,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(n_models)
        ])

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all ensemble members and return aggregate predictions.

        Parameters
        ----------
        state : torch.Tensor
            Current state, shape ``(batch, state_dim)``.
        action : torch.Tensor
            Action, shape ``(batch,)`` or ``(batch, 1)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - **ensemble_mean**: Mean of individual model means, ``(batch, output_dim)``.
            - **aleatoric_var**: Mean of individual model variances (data noise).
            - **epistemic_var**: Variance of individual model means (model uncertainty).
        """
        all_means = []
        all_vars = []

        for model in self.models:
            mean, logvar = model(state, action)
            all_means.append(mean)
            all_vars.append(torch.exp(logvar))

        # Stack: (n_models, batch, output_dim)
        means_stack = torch.stack(all_means, dim=0)
        vars_stack = torch.stack(all_vars, dim=0)

        # Ensemble mean prediction
        ensemble_mean = means_stack.mean(dim=0)

        # Aleatoric uncertainty: average of predicted variances
        aleatoric_var = vars_stack.mean(dim=0)

        # Epistemic uncertainty: variance of predicted means
        epistemic_var = means_stack.var(dim=0)

        return ensemble_mean, aleatoric_var, epistemic_var

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of individual model losses for joint training.

        Parameters
        ----------
        state, action, next_state : torch.Tensor
            Training batch.

        Returns
        -------
        torch.Tensor
            Scalar total loss (sum of all members).
        """
        total_loss = torch.tensor(0.0, device=state.device)
        for model in self.models:
            total_loss = total_loss + model.loss(state, action, next_state)
        return total_loss / self.n_models

    def total_uncertainty(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Total predictive uncertainty (aleatoric + epistemic).

        Returns
        -------
        torch.Tensor
            Total variance, shape ``(batch, output_dim)``.
        """
        _, aleatoric, epistemic = self.forward(state, action)
        return aleatoric + epistemic
