import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any

from .. import BaseTrainableModule
from .scheduler import GaussianNoiseScheduler


class BaseDiffusionModule(BaseTrainableModule, ABC):
    """
    Abstract Base Class for Diffusion Models.

    It implements the standard DDPM training loop (Forward Process -> MSE Loss),
    which is common across almost all diffusion architectures (CNN, Transformer, etc.).

    Subclasses must implement:
    - _build_backbone(): To instantiate the specific neural network.
    """

    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            timesteps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            lr: float = 1e-3,
    ):
        super().__init__()

        # Save common config
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr

        # 1. Physics (Standard Gaussian Schedule)
        # Almost all models use this, so we define it in the base.
        self.scheduler = GaussianNoiseScheduler(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        # 2. Backbone (Abstract)
        # This is the variable part. The base class doesn't know if it's a CNN or Transformer.
        self.backbone = self._build_backbone()

        # 3. Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """
        Factory method to instantiate the neural network (Backbone).
        Must be implemented by subclasses.
        """
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Pass-through to the backbone."""
        return self.backbone(x, t, cond)

    def _parse_batch (self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method that defines custom behavior to adapt standard dataloader batch (of shape B, L, C)
        to backbone implementation needs.
        Base implementation returns tensors of shape (B, C, L).

        Expects batch to be a tuple containing y and c tensors
        """
        if len(batch) != 2:
            raise ValueError(
                f"Batch format error: Expected strictly 2 tensors (Target, Condition), "
                f"but received {len(batch)} elements. "
                f"Check your WindowBuilder/Loader configuration."
            )

        y, c = batch

        y = y.transpose(1, 2)
        c = c.transpose(1, 2)

        return y, c

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
        STANDARD DDPM TRAINING LOOP.
        This logic is 'standardized' and rarely changes.
        """
        x_real, c_cond = self._parse_batch(batch)
        batch_size = x_real.shape[0]
        device = x_real.device

        # 1. Sample random timesteps
        t = self.scheduler.sample_random_timesteps(batch_size, device)

        # 2. Create target noise
        noise = torch.randn_like(x_real)

        # 3. Add noise (Forward Process)
        x_noisy = self.scheduler.add_noise(x_real, noise, t)

        # 4. Optimization Step
        self.optimizer.zero_grad()

        # Predict noise (using the abstract backbone)
        noise_pred = self.backbone(x_noisy, t, c_cond)

        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Standard validation loop."""
        x_real, c_cond = batch
        batch_size = x_real.shape[0]
        device = x_real.device

        t = self.scheduler.sample_random_timesteps(batch_size, device)
        noise = torch.randn_like(x_real)
        x_noisy = self.scheduler.add_noise(x_real, noise, t)

        with torch.no_grad():
            noise_pred = self.backbone(x_noisy, t, c_cond)
            loss = F.mse_loss(noise_pred, noise)

        return {"val_loss": loss.item()}

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, n_samples: int | None = None, verbose: bool = False) -> torch.Tensor:
        """
        Standard DDPM Sampling Loop.
        Uses the scheduler math and the backbone prediction.
        """
        self.eval()
        from tqdm.auto import tqdm
        device = cond.device
        if n_samples is None: n_samples = cond.shape[0]

        # Determine channel count from backbone input layer
        # Assumes backbone has 'input_projection' or similar attribute, or passed via config.
        # A robust way is to use self.input_dim
        img = torch.randn((n_samples, self.input_dim, cond.shape[-1]), device=device)

        iterator = range(self.scheduler.timesteps - 1, -1, -1)
        if verbose: iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            noise_pred = self.backbone(img, t, cond)

            alpha = self.scheduler._extract(self.scheduler.alphas, t, img.shape)
            beta = self.scheduler._extract(self.scheduler.betas, t, img.shape)
            sqrt_one_minus_alpha_cumprod = self.scheduler._extract(
                self.scheduler.sqrt_one_minus_alphas_cumprod, t, img.shape
            )
            mean = (1 / torch.sqrt(alpha)) * (img - (beta / sqrt_one_minus_alpha_cumprod) * noise_pred)
            if i > 0:
                img = mean + torch.sqrt(beta) * torch.randn_like(img)
            else:
                img = mean

        return img

    def get_config(self) -> dict[str, Any]:
        """Base config. Subclasses should extend this."""
        return {
            "input_dim": self.input_dim,
            "cond_dim": self.cond_dim,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "lr": self.lr,
        }