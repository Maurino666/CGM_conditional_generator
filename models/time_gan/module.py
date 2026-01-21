# cgm_models/timegan/module.py

from typing import Any

import torch
from torch import nn, Tensor

from .architecture import (
    Encoder,
    Recovery,
    Generator,
    Supervisor,
    Discriminator,
)
from .. import BaseTrainableModule


def _moment_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """
    Moment-matching loss between real and generated sequences.

    Matches first (mean) and second (std) moments over batch and time
    for each feature, then averages over features.

    Parameters
    ----------
    x : Tensor
        Real sequences, shape (batch_size, seq_len, input_dim).
    x_hat : Tensor
        Generated/reconstructed sequences, same shape as x.

    Returns
    -------
    loss : Tensor
        Scalar tensor representing the moment-matching loss.
    """
    # Collapse batch and time dims, keep feature dim

    real_mean = x.mean(dim=(0, 1))
    real_std = x.std(dim=(0, 1))

    fake_mean = x_hat.mean(dim=(0, 1))
    fake_std = x_hat.std(dim=(0, 1))

    mean_loss = torch.mean(torch.abs(fake_mean - real_mean))
    std_loss = torch.mean(torch.abs(fake_std - real_std))

    return mean_loss + std_loss


class TimeGanModule(BaseTrainableModule):
    """
    TimeGAN training-aware module.

    This class wraps the TimeGAN architecture:
      - Encoder (E)
      - Recovery (R)
      - Generator (G)
      - Supervisor (S)
      - Discriminator (D)

    and exposes high-level training steps:
      - autoencoder_step: pretrain encoder + recovery as an autoencoder
      - supervisor_step: pretrain supervisor to predict H(t+1) from H(t)
      - adversarial_step: full adversarial training (G/S/E/R vs D)

    It expects to receive already-windowed, normalized sequences of shape:
      (batch_size, seq_len, input_dim).
    """

    PHASE_AUTOENCODER = "ae"
    PHASE_SUPERVISOR = "sup"
    PHASE_ADVERSARIAL = "adv"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        noise_dim: int | None = None,
        lr: float = 1e-3,
        beta1: float = 0.5,
        gamma: float = 1.0,
        moment_weight: float = 1.0,
        grad_clip_G: float | None = 1.0,
        grad_clip_D: float | None = 0.5,
        g_steps_per_iter: int = 2,
        d_loss_threshold: float = 0.15,
    ) -> None:
        """
        Initialize the TimeGAN module.

        Parameters
        ----------
        input_dim : int
            Number of features per time step in the original space (X).
        hidden_dim : int
            Dimension of the latent space (H).
        num_layers : int, optional
            Number of GRU layers for all sub-networks, by default 1.
        noise_dim : int | None, optional
            Dimension of the noise space (Z). If None, defaults to input_dim.
        lr : float, optional
            Base learning rate for all optimizers, by default 1e-3.
        beta1 : float, optional
            Beta1 parameter for Adam optimizers, by default 0.5.
        gamma : float, optional
            Weight for the E_hat related adversarial terms (similar to w_gamma).
        moment_weight : float, optional
            Weight for the moment matching terms (similar to w_g).
        """
        super().__init__()

        self.phase = self.PHASE_AUTOENCODER

        # -----------------------------
        # Store hyperparameters
        # -----------------------------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise_dim = noise_dim if noise_dim is not None else input_dim

        self.lr = lr
        self.beta1 = beta1
        self.gamma = gamma
        self.moment_weight = moment_weight
        self.grad_clip_G = grad_clip_G
        self.grad_clip_D = grad_clip_D
        self.g_steps_per_iter = g_steps_per_iter
        self.d_loss_threshold = d_loss_threshold

        # -----------------------------
        # Sub-networks (E, R, G, S, D)
        # -----------------------------
        # NOTE: your Encoder/Recovery/... constructors may take
        #       dimensional arguments instead of an "opt" object.
        self.encoder = Encoder(
            z_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.recovery = Recovery(
            hidden_dim=self.hidden_dim,
            z_dim=self.input_dim,
            num_layers=self.num_layers,
        )
        self.generator = Generator(
            z_dim=self.noise_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.supervisor = Supervisor(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.discriminator = Discriminator(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # -----------------------------
        # Loss functions
        # -----------------------------
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # -----------------------------
        # Optimizers
        # -----------------------------
        self.optimizer_e = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )
        self.optimizer_r = torch.optim.Adam(
            self.recovery.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )
        self.optimizer_s = torch.optim.Adam(
            self.supervisor.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )

    # ------------------------------------------------------------------
    # Helper methods (pure forward logic, no optimization)
    # ------------------------------------------------------------------
    def _encode(self, x: Tensor) -> Tensor:
        """
        Encode original space X into latent space H.

        Parameters
        ----------
        x : Tensor
            Input sequences of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        H : Tensor
            Latent sequences of shape (batch_size, seq_len, hidden_dim).
        """
        H = self.encoder(x)
        return H

    def _recover(self, H: Tensor) -> Tensor:
        """
        Recover / reconstruct X from latent space H.

        Parameters
        ----------
        H : Tensor
            Latent sequences of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        X_tilde : Tensor
            Reconstructed sequences of shape (batch_size, seq_len, input_dim).
        """
        X_tilde = self.recovery(H)
        return X_tilde

    def _supervise(self, H: Tensor) -> Tensor:
        """
        Apply the supervisor network on latent sequences.

        Parameters
        ----------
        H : Tensor
            Latent sequences of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        H_sup : Tensor
            Supervised latent sequences (same shape as H).
        """
        H_sup = self.supervisor(H)
        return H_sup

    def _generate_from_noise(self, Z: Tensor) -> Tensor:
        """
        Map noise sequences Z to latent embeddings E_hat using the generator.

        Parameters
        ----------
        Z : Tensor
            Noise sequences of shape (batch_size, seq_len, noise_dim).

        Returns
        -------
        E_hat : Tensor
            Generated latent sequences of shape (batch_size, seq_len, hidden_dim).
        """
        E_hat = self.generator(Z)
        return E_hat

    def _discriminate(self, H: Tensor) -> Tensor:
        """
        Apply the discriminator on latent sequences H.

        Parameters
        ----------
        H : Tensor
            Latent sequences of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        Y : Tensor
            Discriminator outputs (e.g. probabilities) of shape
            (batch_size, seq_len, 1) or (batch_size, seq_len, hidden_dim),
            depending on your Discriminator design.
        """
        Y = self.discriminator(H)
        return Y

    def _sample_noise_like(self, x: Tensor) -> Tensor:
        """
        Sample Gaussian noise Z with the same (batch, seq_len) as x.

        Parameters
        ----------
        x : Tensor
            Reference tensor with shape (batch_size, seq_len, input_dim).

        Returns
        -------
        Z : Tensor
            Noise tensor of shape (batch_size, seq_len, noise_dim).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        Z = torch.randn(batch_size, seq_len, self.noise_dim, device=device)
        return Z

    # ------------------------------------------------------------------
    # Pretraining steps
    # ------------------------------------------------------------------
    def autoencoder_step(self, x: Tensor) -> float:
        """
        One optimization step for encoder + recovery pretraining.

        This trains the autoencoder E/R to reconstruct the input X.

        Parameters
        ----------
        x : Tensor
            Input batch of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        loss_value : float
            Scalar reconstruction loss value for logging.
        """
        # Forward: X -> H -> X_tilde
        H = self._encode(x)
        X_tilde = self._recover(H)

        # Reconstruction loss
        loss_ae = self.mse_loss(X_tilde, x)

        # Backward on encoder and recovery
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        loss_ae.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.recovery.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_e.step()
        self.optimizer_r.step()

        return float(loss_ae.item())

    def supervisor_step(self, x: Tensor) -> float:
        """
        One optimization step for supervisor pretraining.

        This trains S to predict H(t+1) from H(t) in latent space.

        Parameters
        ----------
        x : Tensor
            Input batch of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        loss_value : float
            Scalar supervised loss value for logging.
        """
        # X -> H
        H = self._encode(x)

        # H -> H_sup
        H_sup = self._supervise(H)

        # Supervised loss on shifted sequences
        # H[:, 1:, :] is H(t+1), H_sup[:, :-1, :] is S(H(t))
        loss_s = self.mse_loss(H[:, 1:, :], H_sup[:, :-1, :])

        self.optimizer_s.zero_grad()
        loss_s.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.supervisor.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_s.step()

        return float(loss_s.item())

    # ------------------------------------------------------------------
    # Adversarial sub-steps
    # ------------------------------------------------------------------

    def _generator_supervisor_step(self, x: Tensor) -> float:
        """
        One optimization step for generator + supervisor (G + S).

        This step:
          - tries to fool the discriminator with generated latent sequences,
          - matches moments (mean/std) between X_hat and X,
          - enforces temporal consistency via the supervised loss in H-space.

        Updates:
          - generator parameters
          - supervisor parameters
        """
        # Real path: X -> H -> H_sup
        H = self._encode(x)
        H_sup = self._supervise(H)

        # Noise path: Z -> E_hat -> H_hat -> X_hat
        Z = self._sample_noise_like(x)
        E_hat = self._generate_from_noise(Z)
        H_hat = self._supervise(E_hat)
        X_hat = self._recover(H_hat)

        # Adversarial losses:
        # Discriminator should think these are real (target = 1)
        Y_fake = self._discriminate(H_hat)
        Y_fake_e = self._discriminate(E_hat)

        # Targets of ones (same shape as outputs)
        ones_like_Y_fake = torch.ones_like(Y_fake)
        ones_like_Y_fake_e = torch.ones_like(Y_fake_e)

        adv_loss_H_hat = self.bce_loss(Y_fake, ones_like_Y_fake)
        adv_loss_E_hat = self.bce_loss(Y_fake_e, ones_like_Y_fake_e)

        # Moment matching between X and X_hat
        moment_loss = _moment_loss(x, X_hat)

        # Supervised loss in latent space
        sup_loss = self.mse_loss(H_sup[:, :-1, :], H[:, 1:, :])

        # Combine all terms (similar spirit to the original implementation)
        loss_g = (
                adv_loss_H_hat
                + self.gamma * adv_loss_E_hat
                + self.moment_weight * moment_loss
                + torch.sqrt(sup_loss + 1e-8)
        )

        # Backprop on G and S
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        loss_g.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.supervisor.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_g.step()
        self.optimizer_s.step()

        return float(loss_g.item())

    def _encoder_recovery_refine_step(self, x: Tensor) -> float:
        """
        Refinement step for encoder + recovery (E + R) with combined loss.

        This mirrors the original backward_er_:

          err_er_ = MSE(X_tilde, X)
          err_s   = MSE(H_supervise(t), H(t+1))
          err_er  = 10 * sqrt(err_er_) + 0.1 * err_s

        Updates:
          - encoder parameters
          - recovery parameters
        """
        # X -> H -> X_tilde
        H = self._encode(x)
        X_tilde = self._recover(H)

        # Supervisory path
        H_sup = self._supervise(H)

        recon_loss = self.mse_loss(X_tilde, x)
        sup_loss = self.mse_loss(H_sup[:, :-1, :], H[:, 1:, :])

        combined_loss = 10.0 * torch.sqrt(recon_loss + 1e-8) + 0.1 * sup_loss

        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        combined_loss.backward()
        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.recovery.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_e.step()
        self.optimizer_r.step()

        return float(combined_loss.item())

    def _discriminator_step(self, x: Tensor) -> float:
        """
        One optimization step for the discriminator (D).

        D learns to classify:
          - H_real (encoder outputs on real X) as real,
          - H_hat (supervised outputs on generated E_hat) as fake,
          - E_hat as fake_e (latent-level adversarial signal).

        Updates only the discriminator parameters.
        """
        # Real latent: X -> H_real (detach to avoid updating encoder here)
        with torch.no_grad():
            H_real = self._encode(x)

        # Fake latent: Z -> E_hat -> H_hat (no gradients to G/S here)
        Z = self._sample_noise_like(x)
        with torch.no_grad():
            E_hat = self._generate_from_noise(Z)
            H_hat = self._supervise(E_hat)

        # Discriminator outputs
        Y_real = self._discriminate(H_real)
        Y_fake = self._discriminate(H_hat)
        Y_fake_e = self._discriminate(E_hat)

        ones_real = torch.ones_like(Y_real)
        zeros_fake = torch.zeros_like(Y_fake)
        zeros_fake_e = torch.zeros_like(Y_fake_e)

        loss_real = self.bce_loss(Y_real, ones_real)
        loss_fake = self.bce_loss(Y_fake, zeros_fake)
        loss_fake_e = self.bce_loss(Y_fake_e, zeros_fake_e)

        loss_d = loss_real + loss_fake + self.gamma * loss_fake_e

        # Optionally skip very small discriminator updates
        if loss_d.item() > self.d_loss_threshold:
            self.optimizer_d.zero_grad()
            loss_d.backward()

            if self.grad_clip_D is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.grad_clip_D)

            self.optimizer_d.step()

        return float(loss_d.item())

    # ------------------------------------------------------------------
    # Adversarial training step (full TimeGAN logic per batch)
    # ------------------------------------------------------------------

    def adversarial_step(self, x: Tensor) -> dict[str, float]:
        """
        One adversarial training cycle on a batch of real sequences.

        Following the original TimeGAN training:
          - run multiple generator+supervisor steps (G + S) and
            encoder+recovery refinement steps (E + R) per batch
          - then update the discriminator once

        x: (batch_size, seq_len, input_dim)

        Returns
        -------
        losses : dict[str, float]
            Dictionary with average losses for this step:
              {
                "g_loss":  ...,
                "er_loss": ...,
                "d_loss":  ...,
              }
        """
        g_losses: list[float] = []
        er_losses: list[float] = []

        # Multiple G/S + ER refinement steps
        for _ in range(self.g_steps_per_iter):
            g_losses.append(self._generator_supervisor_step(x))
            er_losses.append(self._encoder_recovery_refine_step(x))

        # One D step
        d_loss = self._discriminator_step(x)

        avg_g_loss = float(sum(g_losses) / len(g_losses))
        avg_er_loss = float(sum(er_losses) / len(er_losses))

        return {
            "g_loss": avg_g_loss,
            "er_loss": avg_er_loss,
            "d_loss": float(d_loss),
        }

    def set_phase(self, phase: str):
        """
        Set the current training phase for TimeGAN.

        Supported phases:
          - "ae"  : autoencoder pretraining (encoder + recovery)
          - "sup" : supervisor pretraining (supervisor)
          - "adv" : adversarial training (full TimeGAN)

        Parameters
        ----------
        phase : str
            Phase identifier: "ae", "sup", or "adv".

        Raises
        ------
        ValueError
            If an unsupported phase string is provided.
        """
        allowed = {
            self.PHASE_AUTOENCODER,
            self.PHASE_SUPERVISOR,
            self.PHASE_ADVERSARIAL,
        }
        if phase not in allowed:
            raise ValueError(
                f"Unknown TimeGAN phase '{phase}'. "
                f"Allowed phases are: {allowed}"
            )
        self.phase = phase
        print(f"[TimeGAN] Phase set to {self.phase}")

    def training_step(self, x: Tensor) -> Any:
        """
        Run one optimization step for the current TimeGAN phase.

        The behavior depends on ``self.phase``:
          - "ae"  : calls ``autoencoder_step`` (E + R pretraining)
          - "sup" : calls ``supervisor_step`` (S pretraining)
          - "adv" : calls ``adversarial_step`` (full TimeGAN update)

        Each phase-specific method is responsible for forward pass,
        loss computation, backpropagation, and optimizer updates.

        Parameters
        ----------
        x : Tensor
            Input batch of real sequences, shape (batch_size, seq_len, input_dim),
            already moved to the correct device.

        Returns
        -------
        loss_value : float
            Scalar loss value for this batch, used for logging.

        Raises
        ------
        RuntimeError
            If ``self.phase`` is not one of the supported values.
        """
        if self.phase == self.PHASE_AUTOENCODER:
            loss = self.autoencoder_step(x)
            return loss

        elif self.phase == self.PHASE_SUPERVISOR:
            loss = self.supervisor_step(x)
            return loss

        elif self.phase == self.PHASE_ADVERSARIAL:
            losses = self.adversarial_step(x)
            return losses  # dict con g_loss, d_loss, er_loss, ecc.
        # TODO fix: il trainer si aspetta un float

        else:
            raise RuntimeError(
                f"TimeGAN in unknown phase '{self.phase}'. "
                f"Please call set_phase(...) with a valid phase."
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, num_samples: int, seq_len: int) -> Tensor:
        """
        Generate synthetic sequences with the trained TimeGAN model.

        Parameters
        ----------
        num_samples : int
            Number of sequences to generate.
        seq_len : int
            Length of each generated sequence.

        Returns
        -------
        X_hat : Tensor
            Generated sequences of shape (num_samples, seq_len, input_dim),
            in the same normalized space used for training.
        """
        device = next(self.parameters()).device

        # Sample random noise Z
        Z = torch.randn(num_samples, seq_len, self.noise_dim, device=device)

        # Z -> E_hat -> H_hat -> X_hat
        E_hat = self._generate_from_noise(Z)
        H_hat = self._supervise(E_hat)
        X_hat = self._recover(H_hat)

        return X_hat