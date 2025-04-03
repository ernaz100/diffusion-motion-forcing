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

Source code references:
- Original implementation: https://github.com/GuyTevet/motion-diffusion-model/blob/main/diffusion/gaussian_diffusion.py
- Key methods:
  - p_sample_loop: Line 300-350
  - p_sample: Line 250-300
  - p_mean_variance: Line 150-250
  - q_posterior_mean_variance: Line 100-150
"""

import torch
import numpy as np
import math
from enum import Enum
from tqdm import tqdm

class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:ModelMeanType
    Line 30-40 in original implementation.
    
    The model can predict different aspects of the diffusion process:
    - PREVIOUS_X: The model predicts x_{t-1} directly
    - START_X: The model predicts x_0 (the clean data)
    - EPSILON: The model predicts the noise (epsilon)
    """
    PREVIOUS_X = 1  # the model predicts x_{t-1}
    START_X = 2     # the model predicts x_0
    EPSILON = 3     # the model predicts epsilon

class ModelVarType(Enum):
    """
    What is used as the model's output variance.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:ModelVarType
    Line 40-50 in original implementation.
    
    Different ways to handle the variance in the model:
    - LEARNED: The model learns the variance
    - FIXED_SMALL: Use a small fixed variance
    - FIXED_LARGE: Use a large fixed variance
    - LEARNED_RANGE: The model learns a range for the variance
    """
    LEARNED = 1
    FIXED_SMALL = 2
    FIXED_LARGE = 3
    LEARNED_RANGE = 4

class LossType(Enum):
    """
    The type of loss function to use.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:LossType
    Line 50-60 in original implementation.
    
    Different loss functions for training:
    - MSE: Mean squared error
    - RESCALED_MSE: MSE with rescaling
    - KL: Kullback-Leibler divergence
    - RESCALED_KL: KL with rescaling
    """
    MSE = 1
    RESCALED_MSE = 2
    KL = 3
    RESCALED_KL = 4

class NoiseScheduler:
    """
    Utilities for training and sampling diffusion models.
    Original implementation reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion
    Line 60-1000 in original implementation.
    
    This class implements the core diffusion process:
    1. Forward process (q): Gradually adds noise to the data
    2. Reverse process (p): Gradually removes noise from the data
    3. Training: Learns to predict the noise at each timestep
    4. Sampling: Generates new samples by denoising random noise
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
        """
        Initialize the noise scheduler.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.__init__
        Line 100-150 in original implementation.
        
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting value of beta (noise schedule)
            beta_end: Ending value of beta (noise schedule)
            model_mean_type: Type of mean prediction
            model_var_type: Type of variance prediction
            loss_type: Type of loss function
            rescale_timesteps: Whether to rescale timesteps
        """
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
        
        # Additional calculations for reverse process
        # Reciprocal square roots for x_0 prediction
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # Used for reverse process (denoising)
        self.posterior_variance = (
            self.betas[1:] * (1 - self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        )
        # Add an extra element at the beginning to match the number of timesteps
        self.posterior_variance = torch.cat([self.posterior_variance[0:1], self.posterior_variance])
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[0:1], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas[1:] * torch.sqrt(self.alphas_cumprod[:-1]) / (1 - self.alphas_cumprod[1:])
        )
        # Add an extra element at the beginning to match the number of timesteps
        self.posterior_mean_coef1 = torch.cat([self.posterior_mean_coef1[0:1], self.posterior_mean_coef1])
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod[:-1]) * torch.sqrt(self.alphas[1:]) / (1 - self.alphas_cumprod[1:])
        )
        # Add an extra element at the beginning to match the number of timesteps
        self.posterior_mean_coef2 = torch.cat([self.posterior_mean_coef2[0:1], self.posterior_mean_coef2])

    def _scale_timesteps(self, t):
        """
        Scale timesteps for better numerical stability.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion._scale_timesteps
        Line 150-160 in original implementation.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.q_mean_variance
        Line 160-180 in original implementation.
        
        This is the forward process distribution, which tells us how to add noise to the data.
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
        Line 180-200 in original implementation.
        
        This is the forward process, which adds noise to the data according to the noise schedule.
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
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.q_posterior_mean_variance
        Line 200-250 in original implementation.
        
        This is the reverse process distribution, which tells us how to remove noise from the data.
        It's derived from Bayes' rule and the forward process distribution.
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

    def p_sample_loop(
        self,
        model,
        shape,
        text=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        progress=False,
    ):
        """
        Generate samples from the model through iterative denoising.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.
        p_sample_loop
        Line 370-400 in original implementation.

        The process:
        1. Start with random noise (or provided noise)
        2. Iterate backwards through timesteps (T to 0)
        3. At each step, use the model to predict how to denoise the current state
        4. Sample from the predicted distribution to get the next state
        5. Continue until we reach t=0, which should be our clean sample
        
        Args:
            model: The neural network that predicts the denoising direction
            shape: The shape of the output tensor [batch_size, channels, ...]
            text: Optional text conditioning
            noise: Optional starting noise (if None, random noise is used)
            clip_denoised: Whether to clip denoised values to [-1, 1]
            denoised_fn: Optional function to apply to denoised samples
            progress: Whether to show progress bar
            
        Returns:
            The final denoised sample
        """
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Start with random noise if none provided
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
            
        # Create list of timesteps in reverse order (T to 0)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # Iterate through timesteps
        for i in indices:
            # Create tensor of current timestep for each sample in batch
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                # Perform one denoising step
                out = self.p_sample(
                    model,
                    img,
                    t,
                    text=text,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                )
                # Update image with denoised version
                img = out["sample"]
        return img

    def p_sample(
        self,
        model,
        x,
        t,
        text=None,
        clip_denoised=True,
        denoised_fn=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.p_sample
        Line 250-300 in original implementation.
        
        This is a single step in the denoising process:
        1. Get the model's prediction for the mean and variance
        2. Sample from the predicted distribution
        3. Return the sample and predicted x_0
        
        Args:
            model: The neural network that predicts the denoising direction
            x: The current noisy sample x_t
            t: The current timestep
            text: Optional text conditioning
            clip_denoised: Whether to clip denoised values to [-1, 1]
            denoised_fn: Optional function to apply to denoised samples
            
        Returns:
            Dictionary containing:
            - sample: The denoised sample x_{t-1}
            - pred_xstart: The model's prediction of the clean sample x_0
        """
        # Get model's prediction for mean and variance
        out = self.p_mean_variance(
            model,
            x,
            t,
            text=text,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        
        # Generate random noise for sampling
        noise = torch.randn_like(x)
        
        # Create mask to prevent adding noise at t=0
        #When t != 0: mask = 1.0, so we add the noise: mean + 1.0 * sqrt(variance) * noise
        #When t = 0: mask = 0.0, so we don't add noise: mean + 0.0 * sqrt(variance) * noise = mean
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        
        # Sample from the predicted distribution:
        # x_{t-1} = mean + sqrt(variance) * noise
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(
        self,
        model,
        x,
        t,
        text=None,
        clip_denoised=True,
        denoised_fn=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion.
        p_mean_variance
        Line 300-350 in original implementation.

        This function:
        1. Gets the model's output (predicted noise)
        2. Uses the predicted noise to get x_0
        3. Computes the mean and variance of the predicted distribution
        4. Processes the predicted x_0 if needed
        
        Args:
            model: The neural network that predicts the noise
            x: The current noisy sample x_t
            t: The current timestep
            text: Optional text conditioning
            clip_denoised: Whether to clip denoised values to [-1, 1]
            denoised_fn: Optional function to apply to denoised samples
            
        Returns:
            Dictionary containing:
            - mean: The mean of the predicted distribution
            - variance: The variance of the predicted distribution
            - log_variance: The log of the variance
            - pred_xstart: The model's prediction of the clean sample x_0
        """
        # Get batch size and number of channels
        B, C = x.shape[:2]
        assert t.shape == (B,)
            
        # Scale timesteps for better numerical stability
        scaled_t = self._scale_timesteps(t)
        
        # Get model's prediction (noise/epsilon)
        model_output = model(x, scaled_t, text)

        # Handle different model output shapes
        if model_output.shape != x.shape:
            if len(model_output.shape) == 3:
                model_output = model_output.permute(0, 2, 1).unsqueeze(1)
            elif len(model_output.shape) == 3:
                model_output = model_output.unsqueeze(1)

        # Use fixed small variance values from our noise schedule
        model_variance = _extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        # Function to process predicted x_0
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Model predicts noise (epsilon), so we need to:
        # 1. Get x_0 from the predicted noise
        # 2. Use x_0 to get the mean of the posterior distribution
        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
        
        # Compute mean using posterior distribution
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        # Verify shapes match
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        Compute x_0 from x_{t-1}.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion._predict_xstart_from_xprev
        Line 350-360 in original implementation.
        """
        return xprev

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Compute x_0 from epsilon.
        Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:GaussianDiffusion._predict_xstart_from_eps
        Line 360-370 in original implementation.
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D array for a batch of indices.
    Original reference: motion-diffusion-model/diffusion/gaussian_diffusion.py:_extract_into_tensor
    Line 400-420 in original implementation.
    
    This helper function is used to extract values from the noise schedule
    for a batch of timesteps, and broadcast them to the right shape.
    """
    # Handle both numpy arrays and PyTorch tensors
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    else:
        res = arr.to(device=timesteps.device)[timesteps].float()
    
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape) 