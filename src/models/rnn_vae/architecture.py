import torch
from torch import nn
from torch import Tensor


class RnnVae(nn.Module):
    """
    Simple RNN-based Variational Autoencoder for CGM time series.

    Input:  (batch_size, seq_len, input_dim)
    Output: reconstructed sequence with same shape + latent variables.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Choose RNN cell type
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM

        # Encoder RNN maps input sequence -> final hidden state
        self.encoder_rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Latent parameters from final hidden state
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder RNN takes repeated z as input at each time step
        self.decoder_rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Final projection from hidden_dim -> input_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input sequence into latent parameters (mu, logvar).

        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :return:  (mu, logvar) both of shape (batch_size, latent_dim)
        """
        # encoder_rnn returns all hidden states and the last hidden state
        _, h_n = self.encoder_rnn(x)  # h_n: (num_layers, batch_size, hidden_dim)

        # Use last layer's hidden state as summary
        # shape: (batch_size, hidden_dim)
        h_last = h_n[-1]

        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick: sample z ~ N(mu, sigma^2) using epsilon ~ N(0, I).

        :param mu:     Mean of latent distribution, shape (batch_size, latent_dim)
        :param logvar: Log-variance, same shape as mu
        :return:       Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, seq_len: int) -> Tensor:
        """
        Decode latent variable z into a reconstructed sequence.

        :param z:       Latent tensor (batch_size, latent_dim)
        :param seq_len: Target sequence length to reconstruct
        :return:        Reconstructed sequence (batch_size, seq_len, input_dim)
        """
        # batch_size = z.size(0)

        # Repeat z across time dimension
        # shape becomes (batch_size, seq_len, latent_dim)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)

        # Run through decoder RNN
        decoder_outputs, _ = self.decoder_rnn(z_repeated)

        # Map hidden states to output dimension
        # output shape: (batch_size, seq_len, input_dim)
        x_recon = self.output_layer(decoder_outputs)
        return x_recon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: encode -> reparameterize -> decode.

        :param x: Input sequence (batch_size, seq_len, input_dim)
        :return: (x_recon, mu, logvar)
        """
        seq_len = x.size(1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, seq_len=seq_len)

        return x_recon, mu, logvar
