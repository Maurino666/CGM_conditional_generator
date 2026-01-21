from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn, Tensor

from ...module_interfaces import BaseTrainableModule
from .architecture import TimeGan


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


class BaseTimeGanModule(BaseTrainableModule, ABC):

    PHASE_AUTOENCODER = "ae"
    PHASE_SUPERVISOR = "sup"
    PHASE_ADVERSARIAL = "adv"

    def __init__(
        self,
        encoder_input_dim: int,
        generator_input_dim: int,
        recovery_output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        noise_dim: int | None = None,
        lr: float = 1e-3,
        beta1: float = 0.5,
        gamma: float = 1.0,
        moment_weight: float = 1.0,
        supervised_weight: float = 1.0,
        grad_clip_G: float | None = 1.0,
        grad_clip_D: float | None = 0.5,
        g_steps_per_iter: int = 2,
        d_loss_threshold: float = 0.15,
        noise_std: float = 0.0,
        soft_label: float = 1.0
    ) -> None:
        super().__init__()

        self.phase = self.PHASE_AUTOENCODER

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise_dim = noise_dim if noise_dim is not None else generator_input_dim
        self.lr = lr
        self.beta1 = beta1
        self.gamma = gamma
        self.moment_weight = moment_weight
        self.supervised_weight = supervised_weight
        self.grad_clip_G = grad_clip_G
        self.grad_clip_D = grad_clip_D
        self.g_steps_per_iter = g_steps_per_iter
        self.d_loss_threshold = d_loss_threshold
        self.noise_std = noise_std
        self.soft_label = soft_label
        self.soft_label = soft_label

        # --- core networks ---
        self.core = TimeGan(
            encoder_input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            generator_input_dim=generator_input_dim,
            recovery_output_dim=recovery_output_dim,
            num_layers=num_layers,
        )

        # --- shared losses ---
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # --- optimizers on submodules ---
        self.optimizer_e = torch.optim.Adam(
            self.core.encoder.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_r = torch.optim.Adam(
            self.core.recovery.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_g = torch.optim.Adam(
            self.core.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_s = torch.optim.Adam(
            self.core.supervisor.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            self.core.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )

    # -- Hooks to implement in the subclasses --
    @abstractmethod
    def _unpack_batch(self, batch: Any) -> dict[str, Tensor]:
        """From raw batch -> canonical dict (e.g. {'x': ..., 'y': ..., 'c': ...})."""
        raise NotImplementedError
    @abstractmethod
    def _build_encoder_input(self, info: dict[str, Tensor]) -> Tensor:
        """Tensor to feed into encoder for the real path."""
        raise NotImplementedError
    @abstractmethod
    def _build_generator_input(self, info: dict[str, Tensor], Z: Tensor) -> Tensor:
        """Tensor to feed into generator (pure Z or [Z, c])."""
        raise NotImplementedError
    @abstractmethod
    def _get_reconstruction_target(self, info: dict[str, Tensor]) -> Tensor:
        """What to compare recovery output against (x or y)."""
        raise NotImplementedError
    @abstractmethod
    def generate(self, *args, **kwargs) -> Tensor:
        """
        Public generation interface.

        Subclasses must implement this method with a concrete and
        type-safe signature.

        The method must call self._generate_from_tensor to get
        the final output tensor from the generator.

        Note: implementation should use torch.no_grad() to avoid
        wasting memory on gradients.
        """
        raise NotImplementedError

    @property
    def should_validate(self) -> bool:
        return False # TODO consider validation in ar and sup phases

    def _get_generator_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """
        Hook to provide an initial hidden state (h_0) for the Generator.
        By default, returns None (standard GRU init).
        Overridden by subclasses to implement Static Variable Initialization.
        """
        return None

    def _get_encoder_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """
        Hook to provide an initial hidden state (h_0) for the Encoder.
        By default, returns None (standard GRU init).
        Overridden by subclasses to implement Static Variable Initialization.
        """
        return None
    def _get_discriminator_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """
        Hook to provide an initial hidden state (h_0) for the Discriminator.
        By default, returns None (standard GRU init).
        Overridden by subclasses to implement Static Variable Initialization.
        """
        return None

    # -- helpers --
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

    # -- Steps --
    # Auto-Encoder Training Step
    def autoencoder_step(self, batch: Any) -> dict[str, Tensor]:
        """
        One optimization step for encoder + recovery pretraining (AE phase).

        This trains E/R to reconstruct a chosen target tensor, which can be:
          - the full input sequence (unconditional case),
          - only the target feature (conditional case).

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader, interpreted by subclass hooks.

        Returns
        -------
        loss_value : dict[str, Tensor}
            Detached reconstruction loss Tensor for logging.
        """
        # Let the subclass interpret the batch
        info: dict[str, Tensor] = self._unpack_batch(batch)

        # Build encoder input and reconstruction target
        x_enc: Tensor = self._build_encoder_input(info)
        target: Tensor = self._get_reconstruction_target(info)

        # Setting initial state
        batch_size = x_enc.shape[0]
        h_e_init = self._get_encoder_initial_state(info, batch_size)

        # Forward: X_enc -> H -> X_tilde (or y_tilde)
        H, _ = self.core.e_forward(x_enc, hidden_state=h_e_init)
        X_tilde, _ = self.core.r_forward(H)

        # Reconstruction loss
        loss_ae = self.mse_loss(X_tilde, target)

        # Backward on encoder and recovery
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        loss_ae.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.core.encoder.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.core.recovery.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_e.step()
        self.optimizer_r.step()

        return {"er/autoencoder_loss" : loss_ae.detach()}

    # Supervisor Training Step
    def supervisor_step(self, batch: Any) -> dict[str, Tensor]:
        """
        One optimization step for supervisor pretraining (SUP phase).

        This trains the supervisor S to predict H(t+1) from H(t)
        in latent space, using encoder inputs built by subclass hooks.

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader.

        Returns
        -------
        loss_value : dict[str, Tensor]
            Detached supervised loss Tensor for logging.
        """
        info: dict[str, Tensor] = self._unpack_batch(batch)
        x_enc: Tensor = self._build_encoder_input(info)

        # Setting initial state
        batch_size = x_enc.shape[0]
        h_e_init = self._get_encoder_initial_state(info, batch_size)

        # X_enc -> H
        H, _ = self.core.e_forward(x_enc, hidden_state=h_e_init)

        # H -> H_sup
        H_sup, _ = self.core.s_forward(H)

        # Supervised loss on shifted sequences:
        #   H[:, 1:, :]   is H(t+1)
        #   H_sup[:, :-1, :] is S(H(t))
        loss_s = self.mse_loss(H_sup[:, :-1, :], H[:, 1:, :])

        self.optimizer_s.zero_grad()
        loss_s.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.core.supervisor.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_s.step()

        return {"sup/supervisor_loss": loss_s.detach()}

    # Adversary Training Step (All nets)

    def _generator_supervisor_step(self, batch: Any) -> Tensor:
        """
        One optimization step for generator + supervisor (G + S).

        This step:
          - tries to fool the discriminator with generated latent sequences,
          - matches moments (mean/std) between generated output and target,
          - enforces temporal consistency via the supervised loss in H-space.

        Updates:
          - generator parameters
          - supervisor parameters

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader, interpreted via subclass hooks.

        Returns
        -------
        loss_value : Tensor
            Detached generator+supervisor loss Tensor for logging.
        """
        info: dict[str, Tensor] = self._unpack_batch(batch)

        # Real path: X_enc -> H_real -> H_sup
        x_enc: Tensor = self._build_encoder_input(info)
        target: Tensor = self._get_reconstruction_target(info)
        batch_size = x_enc.shape[0]

        # Setting initial states
        h_e_init = self._get_encoder_initial_state(info, batch_size)
        h_g_init = self._get_generator_initial_state(info, batch_size)
        h_d_init = self._get_discriminator_initial_state(info, batch_size)

        # Encoder is used as a fixed feature extractor in this step
        with torch.no_grad():
            H_real, _ = self.core.e_forward(x_enc, hidden_state=h_e_init)

        # Supervisor remains trainable: we want gradients for S
        H_sup, _ = self.core.s_forward(H_real)

        # Noise path: Z + cond -> E_hat -> H_hat -> Y_hat
        Z = self._sample_noise_like(x_enc)
        z_input: Tensor = self._build_generator_input(info, Z)



        E_hat, _ = self.core.g_forward(z_input, hidden_state=h_g_init) # h_g_init can be initialized
        H_hat, _ = self.core.s_forward(E_hat)
        Y_hat, _ = self.core.r_forward(H_hat)  # output in target space

        # Adversarial losses: fool D with H_hat and E_hat
        Y_fake, _ = self.core.d_forward(H_hat, hidden_state=h_d_init)
        Y_fake_e, _ = self.core.d_forward(E_hat, hidden_state=h_d_init)

        ones_like_Y_fake = torch.ones_like(Y_fake)
        ones_like_Y_fake_e = torch.ones_like(Y_fake_e)

        adv_loss_H_hat = self.bce_loss(Y_fake, ones_like_Y_fake)
        adv_loss_E_hat = self.bce_loss(Y_fake_e, ones_like_Y_fake_e)

        # Moment matching between target and generated output
        # For unconditional TimeGAN: target == full X
        # For conditional variant:  target == y (glucose) only
        moment_loss = _moment_loss(target, Y_hat)

        # Supervised loss in latent space (S(H_real) ≈ H_real shifted)
        sup_loss = self.mse_loss(H_sup[:, :-1, :], H_real[:, 1:, :])

        # Combined generator+supervisor loss
        loss_g = (
                adv_loss_H_hat
                + self.gamma * adv_loss_E_hat
                + self.moment_weight * moment_loss
                + self.supervised_weight * torch.sqrt(sup_loss + 1e-8)
        )

        # Backprop on G and S only
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        loss_g.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.core.generator.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.core.supervisor.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_g.step()
        self.optimizer_s.step()

        return loss_g.detach()

    def _encoder_recovery_refine_step(self, batch: Any) -> Tensor:
        """
        Refinement step for encoder + recovery (E + R) with combined loss.

        Mirrors the original TimeGAN refinement:

          err_er_ = MSE(output, target)
          err_s   = MSE(S(H(t)), H(t+1))
          err_er  = 10 * sqrt(err_er_) + 0.1 * err_s

        Updates:
          - encoder parameters
          - recovery parameters

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader.

        Returns
        -------
        loss_value : Tensor
            Detached combined loss Tensor for logging.
        """
        info: dict[str, Tensor] = self._unpack_batch(batch)
        x_enc: Tensor = self._build_encoder_input(info)
        target: Tensor = self._get_reconstruction_target(info)

        # Setting hidden state
        batch_size = x_enc.shape[0]
        h_e_init = self._get_encoder_initial_state(info, batch_size)

        # X_enc -> H -> output
        H, _ = self.core.e_forward(x_enc, hidden_state=h_e_init)
        output, _ = self.core.r_forward(H)

        # Supervisory path (S not updated, but still differentiable)
        H_sup, _ = self.core.s_forward(H)

        recon_loss = self.mse_loss(output, target)
        sup_loss = self.mse_loss(H_sup[:, :-1, :], H[:, 1:, :])

        combined_loss = 10.0 * torch.sqrt(recon_loss + 1e-8) + 0.1 * sup_loss

        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        combined_loss.backward()

        if self.grad_clip_G is not None:
            nn.utils.clip_grad_norm_(self.core.encoder.parameters(), max_norm=self.grad_clip_G)
            nn.utils.clip_grad_norm_(self.core.recovery.parameters(), max_norm=self.grad_clip_G)

        self.optimizer_e.step()
        self.optimizer_r.step()

        return combined_loss.detach()

    def _discriminator_step(self, batch: Any) -> Tensor:
        """
        One optimization step for the discriminator (D).

        D learns to classify:
          - real latent sequences (from encoder) as real,
          - generated + supervised latent sequences as fake,
          - purely generated latent sequences as fake_e.

        Updates only the discriminator parameters.

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader.

        Returns
        -------
        loss_value : Tensor
            Detached discriminator loss Tensor for logging.
        """
        info: dict[str, Tensor] = self._unpack_batch(batch)
        x_enc: Tensor = self._build_encoder_input(info)
        batch_size = x_enc.shape[0]

        # Setting hidden state
        h_e_init = self._get_encoder_initial_state(info, batch_size)
        h_g_init = self._get_generator_initial_state(info, batch_size)
        h_d_init = self._get_discriminator_initial_state(info, batch_size)

        # Real latent: X_enc -> H_real (no grad to encoder here)
        with torch.no_grad():
            H_real, _ = self.core.e_forward(x_enc, hidden_state=h_e_init)

        # Fake latent: Z + cond -> E_hat -> H_hat (no grad to G/S/E here)
        Z = self._sample_noise_like(x_enc)
        z_input: Tensor = self._build_generator_input(info, Z)

        with torch.no_grad():
            E_hat, _ = self.core.g_forward(z_input, hidden_state=h_g_init)
            H_hat, _ = self.core.s_forward(E_hat)

        # Optional Noise injection
        if self.phase == self.PHASE_ADVERSARIAL and self.noise_std > 0:
            H_real = H_real + torch.randn_like(H_real) * self.noise_std
            H_hat = H_hat + torch.randn_like(H_hat) * self.noise_std
            E_hat = E_hat + torch.randn_like(E_hat) * self.noise_std

        # Discriminator outputs
        Y_real, _ = self.core.d_forward(H_real, hidden_state=h_d_init)
        Y_fake, _ = self.core.d_forward(H_hat, hidden_state=h_d_init)
        Y_fake_e, _ = self.core.d_forward(E_hat, hidden_state=h_d_init)

        ones_real = torch.ones_like(Y_real) * self.soft_label
        zeros_fake = torch.zeros_like(Y_fake)
        zeros_fake_e = torch.zeros_like(Y_fake_e)

        loss_real = self.bce_loss(Y_real, ones_real)
        loss_fake = self.bce_loss(Y_fake, zeros_fake)
        loss_fake_e = self.bce_loss(Y_fake_e, zeros_fake_e)

        loss_d = loss_real + loss_fake + self.gamma * loss_fake_e

        # Optional small-loss threshold
        if loss_d.item() > self.d_loss_threshold:
            self.optimizer_d.zero_grad()
            loss_d.backward()

            if self.grad_clip_D is not None:
                nn.utils.clip_grad_norm_(self.core.discriminator.parameters(), max_norm=self.grad_clip_D)

            self.optimizer_d.step()

        return loss_d.detach()


    def adversarial_step(self, batch: Any) -> dict[str, Tensor]:
        """
        One adversarial training cycle on a batch of real sequences.

        Following the original TimeGAN training:
          - run multiple generator+supervisor steps (G + S) and
            encoder+recovery refinement steps (E + R) per batch,
          - then update the discriminator once.

        Parameters
        ----------
        batch : Any
            Raw batch from the DataLoader.

        Returns
        -------
        losses : dict[str, Tensor]
            Dictionary with detached average losses Tensors for this step:
              {
                "g_loss":  ...,
                "er_loss": ...,
                "d_loss":  ...,
              }
        """
        g_losses: list[Tensor] = []
        er_losses: list[Tensor] = []

        # Multiple G/S + ER refinement steps
        for _ in range(self.g_steps_per_iter):
            g_losses.append(self._generator_supervisor_step(batch))
            er_losses.append(self._encoder_recovery_refine_step(batch))

        # One D step
        d_loss = self._discriminator_step(batch)

        avg_g_loss = torch.stack(g_losses).mean()
        avg_er_loss = torch.stack(er_losses).mean()

        return {
            "adv/g_loss": avg_g_loss,
            "adv/er_loss": avg_er_loss,
            "adv/d_loss": d_loss,
        }

    # Phase-wise Training Step compatible with trainer class

    def training_step(self, batch: Any) -> Any:
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
        batch : Any
            Raw batch from the DataLoader (Tensor or tuple, depending
            on the subclass and dataset).

        Returns
        -------
        loss_value_or_dict : float | dict[str, float]
            Scalar loss (AE/SUP) or dict of losses (ADV) for logging.

        Raises
        ------
        RuntimeError
            If ``self.phase`` is not one of the supported values.
        """
        if self.phase == self.PHASE_AUTOENCODER:
            return self.autoencoder_step(batch)

        if self.phase == self.PHASE_SUPERVISOR:
            return self.supervisor_step(batch)

        if self.phase == self.PHASE_ADVERSARIAL:
            return self.adversarial_step(batch)

        raise RuntimeError(
            f"TimeGAN in unknown phase '{self.phase}'. "
            f"Please call set_phase(...) with a valid phase."
        )


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
        print(f"Training Phase set to {self.phase}")

    # Generation
    def _generate_from_tensor(self, generator_input: Tensor, hidden_state: Tensor | None = None) -> Tensor:
        """
        Apply G -> S -> R to a generator input sequence.

        `generator_input` can be pure noise (unconditional case) or a
        concatenation of noise and conditioning features. The last
        dimension must match the generator input size.
        """
        E_hat, _ = self.core.g_forward(generator_input, hidden_state=hidden_state)
        H_hat, _ = self.core.s_forward(E_hat)
        output, _ = self.core.r_forward(H_hat)
        return output