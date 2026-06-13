"""

The module contains:
1. ``MultiClassMA``: custom likelihood for multi-class classification from
   multiple annotators using an instance-dependent reliability term and a
   focal-loss-based variational objective.
2. ``MultitaskGPModel``: sparse variational multi-output GP with an LMC
   variational strategy.
3. ``build_ccgpma``: convenience function that creates a compatible model and
   likelihood pair.

Expected targets
----------------
Targets must have shape ``[batch_size, num_annotators]``. Observed class labels
are encoded from ``0`` to ``num_classes - 1``. Negative values represent
missing annotations and are ignored by the expected log-likelihood.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal, base_distributions
from gpytorch.likelihoods import Likelihood
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D


class MultiClassMA(Likelihood):
    """Custom likelihood for multi-class learning from multiple annotators.

    Parameters
    ----------
    num_classes:
        Number of ground-truth classes.
    num_ann:
        Number of annotators.
    focal_gamma:
        Focusing parameter used in the focal-loss term.
    num_likelihood_samples:
        Number of Monte Carlo samples used for the softmax expectation.
    num_quadrature_points:
        Number of Gauss-Hermite quadrature points used for annotator
        reliability expectations.
    """

    def __init__(
        self,
        num_classes: int,
        num_ann: int,
        focal_gamma: float = 0.1,
        num_likelihood_samples: int = 200,
        num_quadrature_points: int = 20,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if num_ann < 1:
            raise ValueError("num_ann must be at least 1.")
        if num_likelihood_samples < 1:
            raise ValueError("num_likelihood_samples must be positive.")

        self.K = int(num_classes)
        self.R = int(num_ann)
        self.focal_gamma = float(focal_gamma)
        self.num_likelihood_samples = int(num_likelihood_samples)
        self.quadrature = GaussHermiteQuadrature1D(num_quadrature_points)

    def forward(self, function_samples: Tensor):
        """Not used directly because inference is implemented in ``marginal``."""
        return None

    def expected_log_prob(
        self,
        target: Tensor,
        input: MultitaskMultivariateNormal,
    ) -> Tensor:
        print("E_log_prob")

        # One-hot representation for multi-labelers' annotations.
        Y = torch.transpose(torch.nn.functional.one_hot(target.long()), 1, 2)

        # Distribution related to the Ground truth.
        Dk = input[:, :self.K]

        # Distribution related to the annotators' reliabilities.
        Dr = input[:, self.K:]

        # E_{q(f_{1,n})...q(f_{K,n})}[log zeta]
        zeta_samples = self._draw_likelihood_samples_zeta(Dk, 200)

        # numerical stability
        eps = 1e-6
        zeta_samples = torch.clip(zeta_samples, min=eps, max=1 - eps)
        alpha = self.focal_gamma
        E_zeta_log = (
            ((1 - zeta_samples) ** alpha) * torch.log(zeta_samples)
        ).mean(dim=0)
        E_zeta_log = E_zeta_log.unsqueeze(-1).repeat(1, 1, self.R)

        # Focal Loss term
        fl = torch.sum(Y * E_zeta_log, dim=1)

        # E_{q(f_{K+1,n})...q(f_{J,n})}[z_n^r]
        E_z = self.quadrature(torch.nn.functional.sigmoid, Dr)

        return torch.sum(
            (E_z * fl - (1 - E_z) * torch.log(torch.tensor(self.K))),
            dim=1,
        )

    def marginal(self, function_dist: MultitaskMultivariateNormal):
        # Distribution related to the Ground truth.
        Dk = function_dist[:, :self.K]

        # Distribution related to the annotators' reliabilities.
        Dr = function_dist[:, self.K:]

        # Ground truth predictions
        zeta_samples = self._draw_likelihood_samples_zeta(Dk, 200)
        E_zeta = zeta_samples.mean(dim=0)
        E_zeta_2 = torch.square(zeta_samples).mean(dim=0)

        # Reliabilities predictions
        E_z = self.quadrature(torch.nn.functional.sigmoid, Dr)

        Sig_2 = lambda x: torch.square(torch.nn.functional.sigmoid(x))
        E_z2 = self.quadrature(Sig_2, Dr)

        return (
            torch.cat((E_zeta, E_z), 1),
            torch.cat((E_zeta_2 - E_zeta**2, E_z2 - E_z**2), 1),
        )

    def _draw_likelihood_samples_zeta(
        self,
        function_dist: MultitaskMultivariateNormal,
        num_likelihood_samples,
    ) -> Tensor:
        sample_shape = torch.Size(
            [num_likelihood_samples]
            + [1]
            * (
                self.max_plate_nesting
                - len(function_dist.batch_shape)
                - 1
            )
        )

        if self.training:
            num_event_dims = len(function_dist.event_shape)

            function_dist = base_distributions.Normal(
                function_dist.mean,
                function_dist.variance.sqrt(),
            )
            function_dist = base_distributions.Independent(
                function_dist,
                num_event_dims - 1,
            )

        function_samples = function_dist.rsample(sample_shape)
        print(function_samples.shape)

        return torch.nn.functional.softmax(function_samples, dim=2)

    def _draw_likelihood_samples_lam(
        self,
        function_dist: MultitaskMultivariateNormal,
        num_likelihood_samples,
    ) -> Tensor:
        sample_shape = torch.Size(
            [num_likelihood_samples]
            + [1]
            * (
                self.max_plate_nesting
                - len(function_dist.batch_shape)
                - 1
            )
        )

        if self.training:
            num_event_dims = len(function_dist.event_shape)

            function_dist = base_distributions.Normal(
                function_dist.mean,
                function_dist.variance.sqrt(),
            )
            function_dist = base_distributions.Independent(
                function_dist,
                num_event_dims - 1,
            )
            function_samples = function_dist.rsample(sample_shape)

        return torch.nn.functional.sigmoid(function_samples)


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """Sparse variational multi-output GP with an LMC strategy.

    Parameters
    ----------
    num_latents:
        Number of latent GP functions.
    num_tasks:
        Number of model outputs. For CCGPMA this is normally ``K + R``.
    inducing_p:
        Number of inducing points per latent function.
    input_dim:
        Number of input dimensions.
    initial_lengthscale:
        Initial RBF lengthscale.
    inducing_points:
        Optional tensor with shape ``[num_latents, inducing_p, input_dim]``.
    """

    def __init__(
        self,
        num_latents: int,
        num_tasks: int,
        inducing_p: int,
        input_dim: int = 1,
        initial_lengthscale: float = 0.01,
    ) -> None:
        inducing_points = torch.rand(
            num_latents,
            inducing_p,
            input_dim,
        )

        variational_distribution = (
            gpytorch.variational.TrilNaturalVariationalDistribution(
                inducing_points.size(-2),
                batch_shape=torch.Size([num_latents]),
            )
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than
        # a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        )

        # self.mean_module = gpytorch.means.ZeroMean(
        #     batch_shape=torch.Size([num_latents])
        # )

        self.covar_module = gpytorch.kernels.RBFKernel(
            batch_shape=torch.Size([num_latents])
        )

        self.covar_module.lengthscale = torch.tensor(
            [initial_lengthscale] * num_latents
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with
        # each output dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_ccgpma(
    num_classes: int,
    num_ann: int,
    inducing_p: int = 10,
    num_latents: Optional[int] = None,
    input_dim: int = 1,
    initial_lengthscale: float = 0.01,
    focal_gamma: float = 0.1,
    num_likelihood_samples: int = 200,
    num_quadrature_points: int = 20,
) -> Tuple[MultitaskGPModel, MultiClassMA]:
    """Create a compatible CCGPMA model and custom likelihood."""
    num_tasks = num_classes + num_ann

    if num_latents is None:
        num_latents = num_tasks

    model = MultitaskGPModel(
        num_latents=num_latents,
        num_tasks=num_tasks,
        inducing_p=inducing_p,
        input_dim=input_dim,
        initial_lengthscale=initial_lengthscale,
    )

    likelihood = MultiClassMA(
        num_classes=num_classes,
        num_ann=num_ann,
        focal_gamma=focal_gamma,
        num_likelihood_samples=num_likelihood_samples,
        num_quadrature_points=num_quadrature_points,
    )

    return model, likelihood
