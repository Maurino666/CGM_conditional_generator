import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) for Model Parameters.

    This class maintains a "shadow" copy of a model's weights.
    During training, the shadow weights are updated using a moving average
    of the online model's weights.

    Formula:
        param_ema = decay * param_ema + (1 - decay) * param_online

    References:
        - Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging.
        - Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models (DDPM).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model (nn.Module): The online model to track.
            decay (float): The decay factor (beta). Closer to 1.0 means smoother but slower adaptation.
                           Standard values: 0.999, 0.9999.
        """
        super().__init__()
        self.decay = decay

        # Create a deep copy of the model to serve as the shadow model.
        # deepcopy ensures architecture and current initialization are preserved.
        self.ema_model = deepcopy(model)

        # Freeze the EMA model. It is NOT trained via backpropagation.
        # It is only updated via the update() method.
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Updates the EMA weights using the current online model weights.

        Args:
            model (nn.Module): The online model containing the latest gradients/updates.
        """
        # Ensure the shadow model is on the same device as the online model
        # (Lazy device moving handles cases where model is moved to GPU after init)
        online_device = next(model.parameters()).device

        # Iterate over both models simultaneously
        for ema_param, online_param in zip(self.ema_model.parameters(), model.parameters()):

            # Move EMA param to correct device if necessary
            if ema_param.device != online_device:
                ema_param.data = ema_param.data.to(online_device)

            # Apply the EMA formula in-place
            # new_average = decay * old_average + (1 - decay) * current_value
            ema_param.data.mul_(self.decay).add_(online_param.data, alpha=(1 - self.decay))

    def state_dict(self):
        """Returns the state dictionary of the shadow model."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the state dictionary into the shadow model."""
        self.ema_model.load_state_dict(state_dict)