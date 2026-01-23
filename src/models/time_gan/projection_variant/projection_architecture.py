import torch
import torch.nn as nn
from torch import Tensor
from ..base_time_gan.architecture import TimeGan, weights_init


class ProjectionDiscriminator(nn.Module):
    """
    Discriminator implementation that uses 'Projection' instead of concat.

    Computes: f(x, y) = y^T * V * phi(x) + psi(phi(x))
    Where:
      - x is the latent feature (H)
      - y is the condition (c)
    """

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()

        # 1. Feature Extractor (phi): Processes only the latent feature H
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # 2. Linear layer for unconditioned term (psi)
        self.linear_uncond = nn.Linear(hidden_dim, 1)

        # 3. Linear layer for the projection (V)
        self.linear_cond = nn.Linear(cond_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, h_input: Tensor, condition: Tensor, hidden_state: Tensor | None = None):
        """
        Args:
            h_input: Latent sequence (Batch, Seq, Hidden)
            condition: Condition sequence (Batch, Seq, Cond_Dim)
            hidden_state: Initial hidden state for injection.
        """
        # A. Feature Extraction (phi)
        d_outputs, h_new = self.rnn(h_input, hidden_state)
        d_outputs = self.norm(d_outputs)  # (Batch, Seq, Hidden)

        # B. Unconditioned Term (psi) -> "Realism" of the target
        out_uncond = self.linear_uncond(d_outputs)  # (Batch, Seq, 1)

        # C. Conditioned Term (Projection) -> "Alignment" to the conditional features
        # Projects the condition to the latent space
        cond_embedding = self.linear_cond(condition)  # (Batch, Seq, Hidden)

        # Dot product
        out_cond = torch.sum(d_outputs * cond_embedding, dim=-1, keepdim=True)  # (Batch, Seq, 1)

        # D. Sum and Sigmoid
        final_output = self.sigmoid(out_uncond + out_cond)

        return final_output, h_new


class ProjectedTimeGan(TimeGan):
    """
    Extension of TimeGan that replaces Discriminator with ProjectionDiscriminator.
    """

    def __init__(
            self,
            encoder_input_dim: int,
            hidden_dim: int,
            generator_input_dim: int,
            # discriminator_input_dim # Not used anymore
            discriminator_hidden_dim: int,
            cond_dim: int,
            recovery_output_dim: int,
            num_layers: int = 1,
    ) -> None:
        # Init original gan
        super().__init__(
            encoder_input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            generator_input_dim=generator_input_dim,
            discriminator_input_dim=1,  # Dummy value
            recovery_output_dim=recovery_output_dim,
            num_layers=num_layers,
        )

        # Overriding the discriminator module
        self.discriminator = ProjectionDiscriminator(
            input_dim=discriminator_hidden_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def d_forward(self, inputs, hidden_state=None):
        """
        d_forward override.

        'inputs' is a tuple.
        """
        h_input, condition = inputs
        return self.discriminator(h_input, condition, hidden_state)