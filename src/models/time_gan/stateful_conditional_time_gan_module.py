import torch

from torch import Tensor

# Assicurati di importare la classe base corretta dal tuo progetto
from .conditional_time_gan_module import ConditionalTimeGanModule


class StatefulConditionalTimeGanModule(ConditionalTimeGanModule):
    """
    Extension of ConditionalTimeGanModule that supports stateful generation
    for long sequences.

    This model uses the hidden states of the RNNs (Generator, Supervisor, Recovery)
    to maintain temporal continuity across consecutive time windows (chunks).
    """

    def generate_long_sequence(
            self,
            long_cond_seq: Tensor,
            chunk_size: int
    ) -> Tensor:
        """
        Generate a long synthetic sequence by stitching together smaller chunks,
        propagating the hidden states between them to ensure continuity.

        Args:
            long_cond_seq (Tensor): The full conditioning sequence.
                Shape: (batch_size, total_length, cond_dim)
            chunk_size (int): The sequence length the model was trained on (e.g., 24).
                The long sequence will be split into chunks of this size.

        Returns:
            Tensor: The generated long sequence.
                Shape: (batch_size, total_length, 1)
        """
        # Input validation
        if long_cond_seq.ndim != 3:
            raise ValueError(f"long_cond_seq must be (batch, len, dim). Got {long_cond_seq.shape}")

        batch_size, total_len, _ = long_cond_seq.shape
        device = next(self.parameters()).device
        long_cond_seq = long_cond_seq.to(device)

        # 1. Split the long conditioning sequence into chunks
        # use torch.split to handle cases where total_len is not perfectly divisible
        cond_chunks = torch.split(long_cond_seq, chunk_size, dim=1)

        generated_chunks: list[Tensor] = []

        # 2. Initialize hidden states to None (first chunk starts from 0)
        h_g: Tensor | None = None  # Generator hidden state
        h_s: Tensor | None = None  # Supervisor hidden state
        h_r: Tensor | None = None  # Recovery hidden state

        self.eval()  # Ensure we are in eval mode

        with torch.no_grad():
            for i, c_chunk in enumerate(cond_chunks):
                current_batch_size, current_seq_len, _ = c_chunk.shape

                # 3. Sample noise for the current chunk
                Z = torch.randn(current_batch_size, current_seq_len, self.noise_dim, device=device)

                # 4. Prepare Generator Input: Concat [Noise, Condition]
                z_input = torch.cat([Z, c_chunk], dim=-1)

                # 5. Step-by-step Generation with State Propagation

                # Generator step: Z -> E_hat
                # We pass h_g (previous state) and receive h_g (new state)
                E_hat, h_g = self.core.g_forward(z_input, hidden_state=h_g)

                # Supervisor step: E_hat -> H_hat
                # Propagate supervisor state
                H_hat, h_s = self.core.s_forward(E_hat, hidden_state=h_s)

                # Recovery step: H_hat -> X_hat (Final Output)
                # Propagate recovery state
                X_hat, h_r = self.core.r_forward(H_hat, hidden_state=h_r)

                generated_chunks.append(X_hat)

        # 6. Stitch chunks back together
        long_sequence = torch.cat(generated_chunks, dim=1)

        return long_sequence