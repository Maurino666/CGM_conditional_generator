import torch
import torch.nn as nn


class GaussianNoiseScheduler(nn.Module):
    """
    Gaussian Noise Scheduler for Diffusion Probabilistic Models (DDPM).

    This class manages the forward diffusion process (adding noise) and stores
    the pre-calculated coefficients (alphas, betas) required for both
    training (forward) and sampling (reverse).

    It uses a linear beta schedule by default, which is standard for
    continuous diffusion models.
    """

    def __init__(
            self,
            timesteps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02
    ):
        """
        Args:
            timesteps (int): The number of diffusion steps (T).
            beta_start (float): The starting value of beta (at t=0).
            beta_end (float): The ending value of beta (at t=T).
        """
        super().__init__()
        self.timesteps = timesteps

        # 1. Define the linear schedule for beta (variance of noise)
        # We use register_buffer so these tensors are automatically moved
        # to the correct device (CPU/GPU) alongside the model, but are not
        # treated as trainable parameters.
        self.register_buffer(
            "betas",
            torch.linspace(beta_start, beta_end, timesteps)
        )

        # 2. Calculate alphas
        # alpha_t = 1 - beta_t
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)

        # 3. Calculate cumulative product of alphas (alpha_bar)
        # This allows us to jump directly from x_0 to x_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # 4. Pre-calculate coefficients for the Forward Process equation:
        # q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        # Coefficient for the mean: sqrt(alpha_bar_t)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )

        # Coefficient for the variance: sqrt(1 - alpha_bar_t)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Diffusion Process: q(x_t | x_0).
        Adds noise to the original samples to reach the state at specific timesteps.

        Equation:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            original_samples: (Batch, Channels, Seq_Len) - The clean data (x_0).
            noise: (Batch, Channels, Seq_Len) - Gaussian noise (epsilon).
            timesteps: (Batch,) - The target timestep t for each sample.

        Returns:
            torch.Tensor: The noisy samples (x_t).
        """
        # 1. Extract the coefficients for the specific timesteps requested
        # resulting shape: (Batch, 1, 1) to allow broadcasting
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape)

        # 2. Apply the formula
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod * noise)

        return noisy_samples

    def sample_random_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Helper to sample random timesteps for training.

        Args:
            batch_size (int): Number of samples needed.
            device (torch.device): The device to place the tensor on.

        Returns:
            torch.Tensor: Random integers in range [0, timesteps).
        """
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Helper function to extract values from a 1D tensor 'a' at indices 't',
        and reshape the result to (Batch, 1, 1, ...) to match 'x_shape'.

        Args:
            a: 1D tensor of constants (e.g., sqrt_alphas_cumprod).
            t: 1D tensor of indices (timesteps).
            x_shape: Shape of the data tensor (B, C, L).

        Returns:
            torch.Tensor: Reshaped tensor ready for broadcasting.
        """
        batch_size = t.shape[0]

        # Gather values: out[i] = a[t[i]]
        out = a.gather(-1, t.cpu())  # Move t to CPU for gather if needed, usually safer

        # Move back to correct device
        out = out.to(t.device)

        # Reshape to (Batch, 1, 1) for broadcasting against (Batch, Channels, Seq_Len)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))