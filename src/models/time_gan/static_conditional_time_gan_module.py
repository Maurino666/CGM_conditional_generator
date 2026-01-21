from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

# Adjust this import path based on your actual file structure
from .base_time_gan.module import BaseTimeGanModule


class StaticConditionalTimeGanModule(BaseTimeGanModule):
    """
    TimeGAN implementation with Static Variable Injection via Hidden State Initialization.

    This architecture addresses the 'flat-line mode collapse' often seen when 
    concatenating static variables (Age, Sex, HbA1c) directly to the input 
    at every time step.

    Architecture Logic:
    -------------------
    1. Dynamic Conditions (c_dynamic): 
       Variables that change over time (e.g., Insulin, Carbs) are concatenated 
       step-by-step with the input (for Encoder) or Noise (for Generator).

    2. Static Conditions (c_static):
       Variables that are constant (e.g., Age) are NOT concatenated to the input sequence.
       Instead, they are passed through a projection network (`static_embedding`) 
       to initialize the hidden state ($h_0$) of the GRU/LSTM layers.

    Benefits:
    ---------
    - Prevents the Recurrent Network from learning simple identity mappings of constant values.
    - Forces the GRU to learn the temporal dynamics of the Glucose signal.
    - Contextualizes the generation process from the very first time step.
    """

    def __init__(
            self,
            cond_dim: int,
            static_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            noise_dim: int = 8,
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
        """
        Args:
            cond_dim: Number of dynamic conditional features per time step (e.g., Insulin).
            static_dim: Number of static conditional features per patient (e.g., Age, Sex).
            hidden_dim: Dimension of the hidden state in GRU/LSTM.
            num_layers: Number of Recurrent layers.
            noise_dim: Dimension of the random noise vector Z.
            ... (other standard TimeGAN hyperparameters)
        """

        # 1. Define input dimensions for the Core Networks
        # Encoder Input: Target (1) + Dynamic Conditions (cond_dim)
        # Note: Static variables are NOT part of the input dimension here.
        encoder_input_dim = 1 + cond_dim

        # Generator Input: Noise (noise_dim) + Dynamic Conditions (cond_dim)
        generator_input_dim = noise_dim + cond_dim

        # Recovery Output: Target only (Glucose)
        recovery_output_dim = 1

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
            supervised_weight=supervised_weight,
            grad_clip_G=grad_clip_G,
            grad_clip_D=grad_clip_D,
            g_steps_per_iter=g_steps_per_iter,
            d_loss_threshold=d_loss_threshold,
            noise_std=noise_std,
            soft_label=soft_label,
        )

        self.cond_dim = cond_dim
        self.static_dim = static_dim

        # 2. Static Embedding Network (Projector)
        # Maps static features to the flattened hidden state dimension.
        # Output size: num_layers * hidden_dim
        # We use Tanh to match the typical range of GRU hidden states.
        self.static_embedding = nn.Sequential(
            nn.Linear(static_dim, hidden_dim * num_layers),
            nn.Tanh()
        )

        # Explicit initialization for the embedding layer
        for m in self.static_embedding.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    # =========================================================================
    # Data Unpacking & Input Building
    # =========================================================================

    def _unpack_batch(self, batch: Any) -> dict[str, Tensor]:
        """
        Unpacks the batch returned by the WindowBuilder.
        Expected format: (y, c_dynamic, c_static)
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                # Standard case with static variables
                y, c_dyn, c_stat = batch
                return {"y": y, "c_dyn": c_dyn, "c_stat": c_stat}

            elif len(batch) == 2:
                # Fallback case: No static variables provided by loader.
                # We initialize static context to zeros to prevent crash.
                y, c_dyn = batch
                device = y.device
                batch_size = y.shape[0]
                # Warning: Implicit zero-filling
                c_stat = torch.zeros(batch_size, self.static_dim, device=device)
                return {"y": y, "c_dyn": c_dyn, "c_stat": c_stat}

        raise ValueError(f"Batch format not recognized. Expected tuple of length 3, got {type(batch)}")

    def _build_encoder_input(self, info: dict[str, Tensor]) -> Tensor:
        """
        Constructs input for the Encoder: [Target_t, Dynamic_Cond_t]
        Static variables are NOT concatenated here.
        """
        return torch.cat([info["y"], info["c_dyn"]], dim=-1)

    def _build_generator_input(self, info: dict[str, Tensor], Z: Tensor) -> Tensor:
        """
        Constructs input for the Generator: [Noise_t, Dynamic_Cond_t]
        Static variables are NOT concatenated here.
        """
        return torch.cat([Z, info["c_dyn"]], dim=-1)

    def _get_reconstruction_target(self, info: dict[str, Tensor]) -> Tensor:
        """The model attempts to reconstruct the target variable (Glucose)."""
        return info["y"]

    # =========================================================================
    # Static Variable Injection (Hidden State Initialization)
    # =========================================================================

    def _project_static_to_h0(self, c_stat: Tensor, batch_size: int) -> Tensor:
        """
        Helper method that projects static features into the hidden state shape.

        Transformation: 
        (Batch, Static_Dim) -> (Batch, Layers * Hidden) -> (Layers, Batch, Hidden)
        """
        # 1. Project
        h_flat = self.static_embedding(c_stat)

        # 2. Reshape to (Batch, Num_Layers, Hidden_Dim)
        h_reshaped = h_flat.view(batch_size, self.num_layers, self.hidden_dim)

        # 3. Permute to PyTorch RNN format: (Num_Layers, Batch, Hidden_Dim)
        h_final = h_reshaped.permute(1, 0, 2).contiguous()

        return h_final

    def _get_generator_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """Provides the Generator with the context of the specific patient (Age, etc.)."""
        return self._project_static_to_h0(info["c_stat"], batch_size)

    def _get_encoder_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """Provides the Encoder with the context, improving compression."""
        return self._project_static_to_h0(info["c_stat"], batch_size)

    def _get_discriminator_initial_state(self, info: dict[str, Tensor], batch_size: int) -> Tensor | None:
        """Allows the Discriminator to verify if the curve matches the patient profile."""
        return self._project_static_to_h0(info["c_stat"], batch_size)

    # =========================================================================
    # Public Generation Interface
    # =========================================================================

    def prepare_generation_inputs(self, batch: Any) -> tuple[Tensor, dict[str, Any]]:
        """
        Adapts a raw batch for the generic metrics callback.
        Maps internal batch keys to generate() arguments:
          - batch['c_dyn'] -> generate(cond_dynamic=...)
          - batch['c_stat'] -> generate(cond_static=...)
        """
        # 1. Using _unpack_batch logic
        data_dict = self._unpack_batch(batch)

        # 2. Extracting target
        real_X = data_dict["y"]

        # 3. Preparing the arguments required by self.generate()
        gen_kwargs = {
            "cond_dynamic": data_dict["c_dyn"],
            "cond_static": data_dict["c_stat"]
        }

        return real_X, gen_kwargs

    def generate(self, cond_dynamic: Tensor, cond_static: Tensor) -> Tensor:
        """
        Generate synthetic time-series data conditioned on both dynamic and static variables.

        Args:
            cond_dynamic (Tensor): Time-varying conditions. Shape (Batch, Seq_Len, cond_dim).
            cond_static (Tensor): Static patient attributes. Shape (Batch, static_dim).

        Returns:
            Tensor: Generated sequences. Shape (Batch, Seq_Len, 1).
        """
        # Input Validation
        if cond_dynamic.ndim != 3 or cond_dynamic.shape[-1] != self.cond_dim:
            raise ValueError(f"cond_dynamic shape mismatch. Expected (B, T, {self.cond_dim}), got {cond_dynamic.shape}")

        if cond_static.ndim != 2 or cond_static.shape[-1] != self.static_dim:
            raise ValueError(f"cond_static shape mismatch. Expected (B, {self.static_dim}), got {cond_static.shape}")

        batch_size, seq_len, _ = cond_dynamic.shape
        device = next(self.parameters()).device

        cond_dynamic = cond_dynamic.to(device)
        cond_static = cond_static.to(device)

        # 1. Prepare Initial State (Context)
        h0 = self._project_static_to_h0(cond_static, batch_size)

        # 2. Prepare Dynamic Input (Noise + Dynamic Conditions)
        Z = torch.randn(batch_size, seq_len, self.noise_dim, device=device)
        gen_input = torch.cat([Z, cond_dynamic], dim=-1)

        # 3. Generate using the context
        with torch.no_grad():
            # Pass h0 to the modified base method
            y_hat = self._generate_from_tensor(gen_input, hidden_state=h0)

        return y_hat

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary for logging purposes."""
        base_config = super().get_config() if hasattr(super(), "get_config") else {}
        my_config = {
            "module_type": "StaticConditionalTimeGan",
            "cond_dim": self.cond_dim,
            "static_dim": self.static_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers
        }
        return {**base_config, **my_config}