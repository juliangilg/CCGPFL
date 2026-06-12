"""Reusable GPyTorch components for CCGPFL.

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
        *args,
        **kwargs,
    ) -> Tensor:
        """Compute the focal-loss-based expected log probability.

        ``target`` has shape ``[batch_size, R]``. Negative target values are
        interpreted as missing annotations and do not contribute to the ELBO.
        """
        if target.ndim != 2:
            raise ValueError(
                "target must have shape [batch_size, num_annotators]."
            )
        if target.size(-1) != self.R:
            raise ValueError(
                f"Expected {self.R} annotators, received {target.size(-1)}."
            )

        valid_mask = target.ge(0)
        safe_target = target.long().clamp(min=0, max=self.K - 1)

        # [N, R, K] -> [N, K, R]
        y_one_hot = torch.nn.functional.one_hot(
            safe_target, num_classes=self.K
        ).transpose(1, 2)

        # Latent distributions for the ground truth and annotator reliability.
        d_classes = input[:, : self.K]
        d_reliability = input[:, self.K :]

        # E_q(f)[(1 - softmax(f))^gamma log softmax(f)]
        zeta_samples = self._draw_likelihood_samples_zeta(
            d_classes, self.num_likelihood_samples
        )
        eps = torch.finfo(zeta_samples.dtype).eps
        zeta_samples = torch.clamp(zeta_samples, min=eps, max=1.0 - eps)

        expected_focal_log = (
            ((1.0 - zeta_samples) ** self.focal_gamma)
            * torch.log(zeta_samples)
        ).mean(dim=0)
        expected_focal_log = expected_focal_log.unsqueeze(-1).expand(
            -1, -1, self.R
        )

        focal_term = torch.sum(y_one_hot * expected_focal_log, dim=1)

        # E_q(g_r)[sigmoid(g_r)]
        expected_reliability = self.quadrature(
            torch.nn.functional.sigmoid, d_reliability
        )

        log_num_classes = input.mean.new_tensor(float(self.K)).log()
        log_prob = (
            expected_reliability * focal_term
            - (1.0 - expected_reliability) * log_num_classes
        )

        return torch.sum(log_prob * valid_mask.to(log_prob.dtype), dim=1)

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Return predictive means and variances.

        The first ``K`` columns correspond to ground-truth class probabilities.
        The following ``R`` columns correspond to annotator reliabilities.
        """
        d_classes = function_dist[:, : self.K]
        d_reliability = function_dist[:, self.K :]

        zeta_samples = self._draw_likelihood_samples_zeta(
            d_classes, self.num_likelihood_samples
        )
        expected_zeta = zeta_samples.mean(dim=0)
        expected_zeta_squared = torch.square(zeta_samples).mean(dim=0)

        expected_reliability = self.quadrature(
            torch.nn.functional.sigmoid, d_reliability
        )
        expected_reliability_squared = self.quadrature(
            lambda x: torch.square(torch.nn.functional.sigmoid(x)),
            d_reliability,
        )

        predictive_mean = torch.cat(
            (expected_zeta, expected_reliability), dim=1
        )
        predictive_variance = torch.cat(
            (
                expected_zeta_squared - expected_zeta.square(),
                expected_reliability_squared - expected_reliability.square(),
            ),
            dim=1,
        )
        return predictive_mean, predictive_variance

    def _draw_likelihood_samples_zeta(
        self,
        function_dist: MultitaskMultivariateNormal,
        num_likelihood_samples: int,
    ) -> Tensor:
        """Draw latent samples and apply softmax over the class dimension."""
        function_samples = self._draw_likelihood_samples(
            function_dist, num_likelihood_samples
        )
        return torch.nn.functional.softmax(function_samples, dim=-1)

    def _draw_likelihood_samples_lam(
        self,
        function_dist: MultitaskMultivariateNormal,
        num_likelihood_samples: int,
    ) -> Tensor:
        """Draw latent samples and apply sigmoid to obtain reliabilities."""
        function_samples = self._draw_likelihood_samples(
            function_dist, num_likelihood_samples
        )
        return torch.nn.functional.sigmoid(function_samples)

    def _draw_likelihood_samples(
        self,
        function_dist: MultitaskMultivariateNormal,
        num_likelihood_samples: int,
    ) -> Tensor:
        """Draw reparameterized samples from a multitask latent distribution."""
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
                function_dist.mean, function_dist.variance.sqrt()
            )
            function_dist = base_distributions.Independent(
                function_dist, num_event_dims - 1
            )

        return function_dist.rsample(sample_shape)


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
        inducing_points: Optional[Tensor] = None,
    ) -> None:
        if num_latents < 1 or num_tasks < 1 or inducing_p < 1:
            raise ValueError(
                "num_latents, num_tasks, and inducing_p must be positive."
            )

        if inducing_points is None:
            inducing_points = torch.rand(
                num_latents, inducing_p, input_dim
            )

        variational_distribution = (
            gpytorch.variational.TrilNaturalVariationalDistribution(
                inducing_points.size(-2),
                batch_shape=torch.Size([num_latents]),
            )
        )

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

        batch_shape = torch.Size([num_latents])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)
        self.covar_module.lengthscale = torch.full(
            (num_latents, 1, 1), float(initial_lengthscale)
        )

    def forward(self, x: Tensor):
        """Evaluate the latent GP prior."""
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
