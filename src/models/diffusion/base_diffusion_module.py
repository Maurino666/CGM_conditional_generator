import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any

from ..module_interfaces import BaseTrainableModule
from .scheduler import GaussianNoiseScheduler
from .ema import EMA


class BaseDiffusionModule(BaseTrainableModule, ABC):
    """
    Abstract Base Class for Diffusion Models with EMA support.

    Implements:
    1. Standard DDPM training loop.
    2. Hook-based EMA integration.
    3. Dynamic switching between Online and EMA weights for inference.
    """

    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            timesteps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            lr: float = 1e-3,
            use_ema: bool = True,
            ema_decay: float = 0.999,
    ):
        super().__init__()

        # Save common config
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lr = lr
        self.use_ema = use_ema
        self.ema_decay = ema_decay

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


        # 3. EMA Initialization
        # Initializes Exponential Moving Average Shadow Net, mirroring the backbone
        self.ema: EMA | None = None
        if self.use_ema:
            self.ema = EMA(self.backbone, self.ema_decay)

        # 4. Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """
        Factory method to instantiate the neural network (Backbone).
        Must be implemented by subclasses.
        """
        pass

    @property
    def inference_backbone(self) -> nn.Module:
        """
        Dynamic Getter (Hook) for the backbone used during Inference.

        Logic:
            - If EMA is enabled and initialized -> Returns the Shadow Model (EMA).
            - Otherwise -> Returns the Online Model (Standard).

        This ensures that validation and generation always use the best available weights
        without requiring code changes in the generation loops.
        """
        if self.use_ema and self.ema is not None:
            return self.ema.ema_model
        return self.backbone

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

        return y.transpose(1, 2), c.transpose(1, 2)

    def on_train_batch_end(self):
        """
        Hook executed immediately after the optimizer step.
        Used to update the EMA shadow weights.
        """
        if self.use_ema and self.ema is not None:
            self.ema.update(self.backbone)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """
        STANDARD DDPM TRAINING LOOP.
        This logic is 'standardized' and rarely changes.
        """
        x_real, c_cond = self._parse_batch(batch)
        batch_size = x_real.shape[0]
        device = x_real.device

        # 1. Prepare Data
        t = self.scheduler.sample_random_timesteps(batch_size, device)
        noise = torch.randn_like(x_real)
        x_noisy = self.scheduler.add_noise(x_real, noise, t)

        # 2. Optimization Step
        self.optimizer.zero_grad()

        # Predict noise (using the abstract backbone)
        noise_pred = self.backbone(x_noisy, t, c_cond)

        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        self.optimizer.step()

        # 3. Trigger Hook
        self.on_train_batch_end()

        return {"loss": loss.item()}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Validation step using EMA weights if available."""
        x_real, c_cond = self._parse_batch(batch)
        batch_size = x_real.shape[0]
        device = x_real.device

        t = self.scheduler.sample_random_timesteps(batch_size, device)
        noise = torch.randn_like(x_real)
        x_noisy = self.scheduler.add_noise(x_real, noise, t)

        with torch.no_grad():
            noise_pred = self.inference_backbone(x_noisy, t, c_cond)
            loss = F.mse_loss(noise_pred, noise)

        return {"val_loss": loss.item()}

    def prepare_generation_inputs(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        y_real, c_cond = batch

        device = next(self.parameters()).device
        y_real = y_real.to(device)
        c_cond = c_cond.to(device)

        return y_real, {"cond": c_cond}

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, n_samples: int | None = None, verbose: bool = False) -> torch.Tensor:
        """
        Standard DDPM Sampling Loop.
        Uses the scheduler math and the backbone prediction.
        Uses EMA weights if enabled.
        """
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)

        cond = cond.transpose(1, 2)

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
            # Uses Property
            noise_pred = self.inference_backbone(img, t, cond)

            alpha = self.scheduler.extract(self.scheduler.alphas, t, img.shape)
            beta = self.scheduler.extract(self.scheduler.betas, t, img.shape)
            sqrt_one_minus_alpha_cumprod = self.scheduler.extract(
                self.scheduler.sqrt_one_minus_alphas_cumprod, t, img.shape
            )
            mean = (1 / torch.sqrt(alpha)) * (img - (beta / sqrt_one_minus_alpha_cumprod) * noise_pred)
            if i > 0:
                img = mean + torch.sqrt(beta) * torch.randn_like(img)
            else:
                img = mean

        return img.transpose(1, 2)

    def get_config(self) -> dict[str, Any]:
        """Base config. Subclasses should extend this."""
        return {
            "input_dim": self.input_dim,
            "cond_dim": self.cond_dim,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "lr": self.lr,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay
        }

    def to_checkpoint(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Saves the model state.
        Includes the EMA state dict if EMA is enabled.
        """
        ckpt = super().to_checkpoint(extra)

        if self.use_ema and self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()

        return ckpt

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint: dict[str, Any],
            map_location: str | torch.device | None = None,
    ) -> "BaseDiffusionModule":
        """
        Loads the model from checkpoint.
        Handles loading the EMA state dict alongside the standard model.
        """
        # 1. Instantiate the class and load standard weights (via super)
        model = super().from_checkpoint(checkpoint, map_location)

        # 2. Check if we need to load EMA weights
        # We need to cast 'model' because super() returns BaseTrainableModule
        if isinstance(model, BaseDiffusionModule) and model.use_ema and "ema_state_dict" in checkpoint:
            # model.ema is initialized in __init__ with random weights (via deepcopy).
            # We must overwrite them with the trained EMA weights.
            model.ema.load_state_dict(checkpoint["ema_state_dict"])

            if map_location is not None:
                # Ensure EMA tensors are on the correct device
                model.ema.ema_model.to(map_location)

        return model