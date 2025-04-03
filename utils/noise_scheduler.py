"""
Noise scheduler for diffusion models.
Original implementation reference: motion-diffusion-model/diffusion/gaussian_diffusion.py

The noise scheduler implements the forward diffusion process, which gradually adds noise to the data.
The process follows the equation:
    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
Where:
    - alpha_cumprod_t is the cumulative product of (1 - beta) up to timestep t
    - beta is the noise schedule that increases from beta_start to beta_end
    - x_0 is our clean data
    - x_t is the noisy data at timestep t

Key points:
    1. At t=0, alpha_cumprod_t = 1, so x_t = x_0 (no noise)
    2. At t=T, alpha_cumprod_t ≈ 0, so x_t ≈ noise (completely noisy)
    3. In between, we get a weighted combination of clean data and noise
"""

import torch
import numpy as np
import math
from enum import Enum

class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:ModelMeanType
    """
    PREVIOUS_X = 1  # the model predicts x_{t-1}
    START_X = 2     # the model predicts x_0
    EPSILON = 3     # the model predicts epsilon

class ModelVarType(Enum):
    """
    What is used as the model's output variance.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:ModelVarType
    """
    LEARNED = 1
    FIXED_SMALL = 2
    FIXED_LARGE = 3
    LEARNED_RANGE = 4

class LossType(Enum):
    """
    The type of loss function to use.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:LossType
    """
    MSE = 1
    RESCALED_MSE = 2
    KL = 3
    RESCALED_KL = 4

class NoiseScheduler:
    """
    Utilities for training and sampling diffusion models.
    Original implementation reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    ):
        self.num_timesteps = num_timesteps
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Create noise schedule
        # beta: noise schedule that increases from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # alpha: 1 - beta
        self.alphas = 1 - self.betas
        # alpha_cumprod: cumulative product of alpha up to each timestep
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        # sqrt(alpha_cumprod): coefficient for clean data in noise addition
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - alpha_cumprod): coefficient for noise in noise addition
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # Used for reverse process (denoising)
        self.posterior_variance = (
            self.betas[1:] * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas[1:] * torch.sqrt(self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod[:-1]) * torch.sqrt(self.alphas[1:]) / (1 - self.alphas_cumprod[1:])
        )

    def _scale_timesteps(self, t):
        """
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion._scale_timesteps
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.q_mean_variance
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            torch.log(1.0 - self.alphas_cumprod), t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        Implements the forward diffusion process:
            x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        Where:
            - x_start (x_0): clean data
            - t: timestep
            - noise: random Gaussian noise
            - x_t: noisy data at timestep t
        
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.q_sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        Used for the reverse process (denoising).
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.q_posterior_mean_variance
        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D array for a batch of indices.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:_extract_into_tensor
    """
    # Handle both numpy arrays and PyTorch tensors
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr.to(device=timesteps.device)[timesteps].float()
    
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape) 