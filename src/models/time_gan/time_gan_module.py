import torch
from torch import Tensor

from .base_time_gan.module import BaseTimeGanModule

class TimeGanModule(BaseTimeGanModule):
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
        gen_input_dim = noise_dim if noise_dim is not None else input_dim

        super().__init__(
            encoder_input_dim=input_dim,
            generator_input_dim=gen_input_dim,
            recovery_output_dim=input_dim,  # reconstruct full X
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            noise_dim=gen_input_dim,
            lr=lr,
            beta1=beta1,
            gamma=gamma,
            moment_weight=moment_weight,
            grad_clip_G=grad_clip_G,
            grad_clip_D=grad_clip_D,
            g_steps_per_iter=g_steps_per_iter,
            d_loss_threshold=d_loss_threshold,
        )


    def _unpack_batch(self, batch: Tensor) -> dict[str, Tensor]:
        """
        Interpret the raw batch as a single input tensor X.
        """
        return {"x": batch}

    def _build_encoder_input(self, info: dict[str, Tensor]) -> Tensor:
        """
        Encoder sees the full input X.
        """
        return info["x"]

    def _build_generator_input(self, info: dict[str, Tensor], Z: Tensor) -> Tensor:
        """
        Generator input is pure noise Z.
        """
        return Z

    def _get_reconstruction_target(self, info: dict[str, Tensor]) -> Tensor:
        """
        Target is the original input X.
        """
        return info["x"]

    def generate(self, num_samples: int, seq_len: int) -> Tensor:
        device = next(self.parameters()).device

        Z = torch.randn(num_samples, seq_len, self.noise_dim, device=device)

        with torch.no_grad():
            output = self._generate_from_tensor(Z)

        return output

