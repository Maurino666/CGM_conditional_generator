from typing import Any

import torch
from torch import nn, Tensor

from .architecture import RnnVae
from ..module_interfaces import BaseTrainableModule


class RnnVaeModule(BaseTrainableModule):
    """
    Training-aware wrapper around the RnnVae architecture.

    This module:
    - owns a RnnVae instance (self.model),
    - stores training hyperparameters (beta, lr),
    - exposes:
        * forward(x): standard VAE forward pass
        * training_step(batch): compute training loss for a batch
        * validation_step(batch): compute validation loss for a batch
        * configure_optimizers(): create and return the optimizer

    The trainer can interact with this module without knowing
    VAE-specific details such as mu/logvar or the KL term.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        beta: float = 1.0,
        lr: float = 1e-3,
        grad_clip: float | None = 1.0
    ) -> None:
        """
        Initialize the RnnVaeModule.

        Parameters
        ----------
        input_dim : int
            Number of input features per time step.
        hidden_dim : int
            Hidden dimension of the encoder/decoder RNNs.
        latent_dim : int
            Dimension of the latent space.
        num_layers : int, optional
            Number of RNN layers in encoder and decoder, by default 1.
        rnn_type : str, optional
            RNN cell type, "gru" or "lstm", by default "gru".
        beta : float, optional
            Weight of the KL divergence term in the VAE loss, by default 1.0.
        lr : float, optional
            Learning rate for the optimizer, by default 1e-3.
        grad_clip : float, optional
            Gradient clipping parameter, by default 1.0.
        """
        super().__init__()

        # Store hyperparameters
        self.beta = beta
        self.lr = lr
        self.grad_clip = grad_clip

        # Underlying VAE architecture
        self.model = RnnVae(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
        )
        # Configure optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)



    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Compute VAE loss = reconstruction loss + beta * KL divergence.

        Parameters
        ----------
        x : Tensor
            Original input sequence (batch_size, seq_len, input_dim).
        x_recon : Tensor
            Reconstructed sequence (same shape as x).
        mu : Tensor
            Latent mean (batch_size, latent_dim).
        logvar : Tensor
            Latent log-variance (batch_size, latent_dim).

        Returns
        -------
        loss : Tensor
            Scalar loss tensor (single value).
        """
        # Reconstruction loss (MSE averaged over all elements)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

        # KL divergence term (average over batch)
        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        )
        kl_loss = kl_per_sample.mean()

        return recon_loss + self.beta * kl_loss

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Tensor) -> float:
        """
        Perform one optimization step on a single batch.

        This method:
          - runs a forward pass,
          - computes the VAE loss (reconstruction + KL),
          - performs backpropagation and an optimizer step,
          - returns the scalar loss value for logging.

        The trainer is expected to:
          - move the batch to the correct device,
          - call this method for each batch,
          - aggregate/print the returned loss values.

        Parameters
        ----------
        batch : Tensor
            Input batch of sequences with shape (batch_size, seq_len, input_dim).

        Returns
        -------
        loss_value : float
            Scalar loss value for this batch, detached from the computation graph.
        """
        # Here we assume the batch is already on the correct device
        x = batch  # shape: (batch_size, seq_len, input_dim)

        # Forward pass through the VAE
        x_recon, mu, logvar = self.model(x)

        # Compute VAE loss (reconstruction + KL)
        loss = self._compute_loss(x, x_recon, mu, logvar)

        # Backpropagation and optimizer step
        self.optimizer.zero_grad()

        loss.backward()

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        # Return a plain Python float for easier logging/aggregation
        return float(loss.item())

    def validation_step(self, batch: Tensor) -> Tensor:
        """
        Compute the validation loss for a single batch.

        This method mirrors training_step but is intended to be used
        under torch.no_grad() and without optimizer updates.

        Parameters
        ----------
        batch : Tensor
            Input batch of sequences (batch_size, seq_len, input_dim).

        Returns
        -------
        loss : Tensor
            Scalar loss tensor for this batch.
        """
        x = batch

        x_recon, mu, logvar = self.model(x)
        loss = self._compute_loss(x, x_recon, mu, logvar)
        return loss

    # ------------------------------------------------------------------
    # Optional: sampling interface
    # ------------------------------------------------------------------
    def sample(self, num_samples: int, seq_len: int) -> Tensor:
        """
        Generate synthetic sequences by sampling from the latent prior.

        Parameters
        ----------
        num_samples : int
            Number of sequences to generate.
        seq_len : int
            Length of each generated sequence.

        Returns
        -------
        samples : Tensor
            Generated sequences of shape (num_samples, seq_len, input_dim).
        """
        device = next(self.parameters()).device
        latent_dim = self.model.latent_dim

        # Sample from standard normal prior in latent space
        z = torch.randn(num_samples, latent_dim, device=device)

        # Decode using the underlying VAE
        x_recon = self.model.decode(z, seq_len=seq_len)
        return x_recon

    def get_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "lr": self.lr,
            "grad_clip": self.grad_clip,
        }
