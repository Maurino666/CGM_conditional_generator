from base_time_gan.module import BaseTimeGanModule

import torch
from torch import Tensor


class ConditionalTimeGanModule(BaseTimeGanModule):
    """
    Conditional TimeGAN module.

    The model:
      - encodes [y, c] into a latent space,
      - reconstructs only the target y from the latent representation,
      - generates y sequences conditioned on c by feeding [Z, c] to G.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        noise_dim: int = 8,
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
        Parameters
        ----------
        cond_dim : int
            Number of conditional features per time step (dim(c)).
        hidden_dim : int
            Latent dimension H.
        num_layers : int, optional
            Number of GRU layers in each sub-network.
        noise_dim : int, optional
            Dimension of the noise fed to the generator (dim(Z)).
        """
        encoder_input_dim = 1 + cond_dim          # [y, c]
        generator_input_dim = noise_dim + cond_dim  # [Z, c]
        recovery_output_dim = 1                   # only y

        super().__init__(
            encoder_input_dim=encoder_input_dim,
            generator_input_dim=generator_input_dim,
            recovery_output_dim=recovery_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            noise_dim=noise_dim,
            lr=lr,
            beta1=beta1,
            gamma=gamma,
            moment_weight=moment_weight,
            grad_clip_G=grad_clip_G,
            grad_clip_D=grad_clip_D,
            g_steps_per_iter=g_steps_per_iter,
            d_loss_threshold=d_loss_threshold,
        )

        self.cond_dim = cond_dim

    # Hooks required by BaseTimeGanModule (conditional case)
    def _unpack_batch(self, batch: tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """
        Expect a batch as (y, c) from the DataLoader.
        """
        y, c = batch
        return {"y": y, "c": c}

    def _build_encoder_input(self, info: Dict[str, Tensor]) -> Tensor:
        """
        Encoder sees the concatenation [y, c] along the feature dimension.
        """
        return torch.cat([info["y"], info["c"]], dim=-1)

    def _build_generator_input(self, info: Dict[str, Tensor], Z: Tensor) -> Tensor:
        """
        Generator input is [Z, c] in the conditional case.
        Z has shape (B, T, noise_dim), c has shape (B, T, cond_dim).
        """
        return torch.cat([Z, info["c"]], dim=-1)

    def _get_reconstruction_target(self, info: Dict[str, Tensor]) -> Tensor:
        """
        Reconstruction target is the target sequence y only.
        """
        return info["y"]

    # Public generation API (conditional)
    def generate(self, cond_seq: Tensor) -> Tensor:
        """
        Generate target sequences conditioned on a given conditioning sequence.

        Parameters
        ----------
        cond_seq : Tensor
            Conditioning sequences, shape (batch_size, seq_len, cond_dim).

        Returns
        -------
        y_hat : Tensor
            Generated target sequences, shape (batch_size, seq_len, 1),
            in the same normalized space used for training.
        """
        if cond_seq.ndim != 3 or cond_seq.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond_seq must have shape (batch_size, seq_len, {self.cond_dim}). "
                f"Got {tuple(cond_seq.shape)} instead."
            )

        batch_size, seq_len, _ = cond_seq.shape
        device = next(self.parameters()).device
        cond_seq = cond_seq.to(device)

        # Sample noise matching cond_seq
        Z = torch.randn(batch_size, seq_len, self.noise_dim, device=device)
        generator_input = torch.cat([Z, cond_seq], dim=-1)

        with torch.no_grad():
            y_hat = self._generate_from_tensor(generator_input)

        return y_hat
