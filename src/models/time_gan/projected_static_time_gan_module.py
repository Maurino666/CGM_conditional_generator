import torch
from torch import Tensor

from .projection_variant.projection_architecture import ProjectedTimeGan
from .static_conditional_time_gan_module import StaticConditionalTimeGanModule


class ProjectedStaticTimeGanModule(StaticConditionalTimeGanModule):
    """
    Variant of StaticConditionalTimeGanModule that uses Projection Discriminator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We override the discriminator with the projection variant
        self.core = ProjectedTimeGan(
            encoder_input_dim=1 + self.cond_dim,
            hidden_dim=self.hidden_dim,
            generator_input_dim=self.noise_dim + self.cond_dim,

            # Projection-specific params
            discriminator_hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,

            recovery_output_dim=1,
            num_layers=self.num_layers,
        )

        # Re-Initializing the optimizer
        self.optimizer_d = torch.optim.Adam(
            self.core.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999)
        )

    def _build_discriminator_input(self, info: dict[str, Tensor], input: Tensor) -> tuple[Tensor, Tensor]:
        """
        Override: discriminator now needs a tuple

        Args:
            info: dict with real data
            input: latent tensor H (real or synthetic)
        """
        cond = info["c"]

        # Adding optional noise
        if self.training and self.phase == "adv" and self.d_cond_noise_std > 0:
            cond = cond + torch.randn_like(cond) * self.d_cond_noise_std

        # Now returns a tuple
        return input, cond