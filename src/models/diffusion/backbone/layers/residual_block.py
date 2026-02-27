import math
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    The core component of DiffWave.
    It combines Dilated Convolutions with a Gated Activation Unit (WaveNet style).

    Flow:
    Input -> [Dilated Conv] --+--> [Gate (Sigmoid)]
                              |           * --> [1x1 Conv] -> Residual (Next Layer)
    Condition --[1x1 Conv] ---+--> [Filter (Tanh)]      --> [1x1 Conv] -> Skip (Output)
    Time -------[Linear] -----+
    """

    def __init__(self, residual_channels, dilation, cond_channels, causal = False):
        super().__init__()

        self.causal = causal
        self.dilation = dilation

        if causal:
            # Causal: pad only on the left side.
            # For kernel_size=3 and dilation=d, we need (kernel_size - 1) * dilation
            # total left padding to ensure output[t] depends only on input[<=t].
            self.causal_pad = (3 - 1) * dilation
            conv_padding = 0  # no automatic padding, we handle it manually
        else:
            # Bidirectional: symmetric padding preserves sequence length
            conv_padding = dilation

        # 1. Dilated Convolution
        # We use padding=dilation to ensure the sequence length remains unchanged (Causal padding not strictly needed for generation, but good for consistency)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,  # 2x channels because we split into Filter/Gate later
            kernel_size=3,
            padding=conv_padding,
            dilation=dilation
        )

        # 2. Conditioning Projection (Insulin/Carbs)
        # Maps condition features to the internal space to modulate the signal
        self.condition_projection = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1)

        # 3. Time Projection (Diffusion Step)
        # Maps the time embedding to shift the features
        self.diffusion_projection = nn.Linear(residual_channels, 2 * residual_channels)

        # 4. Output Projections
        self.output_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x, diffusion_embed, conditioner):
        """
        x: (B, Res_Channels, L)
        diffusion_embed: (B, Res_Channels) - Note: Global time info
        conditioner: (B, Cond_Channels, L) - Note: Local condition info
        """
        h = x

        # A. Apply Dilated Convolution to the signal
        h = self.dilated_conv(h)  # (B, 2*C, L)

        # B. Add Conditioning Information (Local)
        # We project the insulin/carbs and add them to the signal
        cond_out = self.condition_projection(conditioner)  # (B, 2*C, L)
        h = h + cond_out

        # C. Add Time Information (Global)
        # We project time and expand dims to broadcast over sequence length
        time_out = self.diffusion_projection(diffusion_embed).unsqueeze(-1)  # (B, 2*C, 1)
        h = h + time_out

        # D. Gated Activation Unit (The "Switch")
        # We split the tensor into two halves:
        # - Filter: The actual feature content (activated by Tanh)
        # - Gate: The control signal (activated by Sigmoid)
        filter_gate, filter_info = h.chunk(2, dim=1)

        # Element-wise multiplication: The Gate decides how much info passes through
        h = torch.tanh(filter_info) * torch.sigmoid(filter_gate)

        # E. Projections
        # Skip connection goes to the final accumulator (collects multi-scale features)
        skip = self.skip_projection(h)

        # Residual connection goes to the next layer (x + transformed_x)
        residual = self.output_projection(h)

        # Important: The residual connection stabilizes deep networks
        return (x + residual) / math.sqrt(2.0), skip