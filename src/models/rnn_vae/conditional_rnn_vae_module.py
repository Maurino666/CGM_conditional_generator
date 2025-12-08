from typing import Any

import torch
from torch import nn, Tensor

from .architecture import RnnVae
from ..module_interfaces import BaseTrainableModule


class ConditionalRnnVaeModule(BaseTrainableModule):
    """
    Conditional RNN-based VAE for CGM time series.

    The model:
      - encodes the concatenation [y, c] into a latent variable z,
      - decodes z into a sequence of the target only (output_dim = 1),
      - uses MSE(y, y_recon) + beta * KL as training loss.

    Expected batch from the DataLoader (same as conditional TimeGAN):
      batch = (y, c)
        y: (batch_size, seq_len, 1)
        c: (batch_size, seq_len, cond_dim)
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        beta: float = 1.0,
        lr: float = 1e-3,
        grad_clip: float | None = 1.0,
    ) -> None:
        super().__init__()

        self.cond_dim = cond_dim
        self.beta = beta
        self.lr = lr
        self.grad_clip = grad_clip

        # Model sees [y, c] as input and outputs only y_hat
        input_dim = 1 + cond_dim
        output_dim = 1

        self.model = RnnVae(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        y: Tensor,
        y_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Conditional VAE loss on the target y only.
        """
        recon_loss = nn.functional.mse_loss(y_recon, y, reduction="mean")

        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )
        kl_loss = kl_per_sample.mean()

        return recon_loss + self.beta * kl_loss

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------
    def training_step(self, batch: tuple[Tensor, Tensor]) -> float:
        """
        One optimization step on a batch (y, c).

        y: (batch_size, seq_len, 1)
        c: (batch_size, seq_len, cond_dim)
        """
        y, c = batch

        if y.ndim != 3 or y.shape[-1] != 1:
            raise ValueError(
                f"y must have shape (batch_size, seq_len, 1), got {tuple(y.shape)}."
            )
        if c.ndim != 3 or c.shape[-1] != self.cond_dim:
            raise ValueError(
                f"c must have shape (batch_size, seq_len, {self.cond_dim}), "
                f"got {tuple(c.shape)}."
            )
        if y.shape[:2] != c.shape[:2]:
            raise ValueError("y and c must share batch_size and seq_len.")

        # Build input [y, c] for the VAE
        x_full = torch.cat([y, c], dim=-1)  # (B, T, 1 + cond_dim)

        # Forward pass through the VAE
        y_recon, mu, logvar = self.model(x_full)

        # Compute VAE loss (reconstruction on y + KL)
        loss = self._compute_loss(y, y_recon, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        return float(loss.item())

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute validation loss on a batch (y, c) without updating weights.
        """
        y, c = batch

        if y.ndim != 3 or y.shape[-1] != 1:
            raise ValueError(
                f"y must have shape (batch_size, seq_len, 1), got {tuple(y.shape)}."
            )
        if c.ndim != 3 or c.shape[-1] != self.cond_dim:
            raise ValueError(
                f"c must have shape (batch_size, seq_len, {self.cond_dim}), "
                f"got {tuple(c.shape)}."
            )
        if y.shape[:2] != c.shape[:2]:
            raise ValueError("y and c must share batch_size and seq_len.")

        x_full = torch.cat([y, c], dim=-1)

        y_recon, mu, logvar = self.model(x_full)
        loss = self._compute_loss(y, y_recon, mu, logvar)
        return loss

    # Sampling interface
    def sample(self, num_samples: int, seq_len: int) -> Tensor:
        """
        Sample from the latent prior and decode to target space.

        Note: this currently ignores any conditioning c and produces
        unconditional samples in the learned target space.
        """
        device = next(self.parameters()).device
        latent_dim = self.model.latent_dim

        z = torch.randn(num_samples, latent_dim, device=device)
        y_hat = self.model.decode(z, seq_len=seq_len)
        return y_hat

    def get_config(self) -> dict[str, Any]:
        """
        Return a serializable configuration dictionary for this conditional RNN-VAE.

        It describes architecture (dims, layers, rnn type) and training hyperparameters.
        """
        return {
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "lr": self.lr,
            "grad_clip": self.grad_clip,
        }
