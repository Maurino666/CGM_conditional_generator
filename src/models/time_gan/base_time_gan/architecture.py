"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen (robinbg@foxmail.com)

-----------------------------

model.py: Network Modules

(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

    Args:
        input_dim (int): Input dimension (e.g. features + condition).
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of RNN layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
    ):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(
            self,
            input: Tensor,
            hidden_state: Tensor | None = None,
    ):
        """
        Forward pass with optional hidden state propagation.

        Args:
            input (torch.Tensor): Input sequence features.
            hidden_state (torch.Tensor, optional): Previous hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - H: Embeddings.
                - h_new: Updated hidden state.
        """
        e_outputs, h_new = self.rnn(input, hidden_state)
        e_outputs = self.norm(e_outputs)
        H = self.fc(e_outputs)
        H = self.tanh(H)
        return H, h_new


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
        hidden_dim (int): Hidden dimension size.
        output_dim (int): Output dimension (original space).
        num_layers (int): Number of RNN layers.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 1,
    ):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first= True)

        #  self.norm = nn.BatchNorm1d(opt.z_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(
            self,
            input: Tensor,
            hidden_state: Tensor | None = None,
    ):
        """
        Forward pass with optional hidden state propagation.

        Args:
            input (torch.Tensor): Latent representation.
            hidden_state (torch.Tensor, optional): Previous hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - X_tilde: Recovered data.
                - h_new: Updated hidden state.
        """
        r_outputs, h_new = self.rnn(input, hidden_state)
        X_tilde = self.fc(r_outputs)
        X_tilde = self.sigmoid(X_tilde)
        return X_tilde, h_new


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
        input_dim (int): Input dimension (noise + condition).
        hidden_dim (int): Hidden dimension size.
        num_layers (int): Number of RNN layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
    ):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(
            self,
            input: Tensor,
            hidden_state: Tensor | None = None,
    ):
        """
        Forward pass with optional hidden state propagation.

        Args:
            input (torch.Tensor): Random variables (concatenated with conditions if applicable).
            hidden_state (torch.Tensor, optional): Previous hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - E: Generated embedding.
                - h_new: Updated hidden state.
        """
        g_outputs, h_new = self.rnn(input, hidden_state)
        g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        E = self.tanh(E)
        return E, h_new


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence in latent space.

    Args:
        hidden_dim (int): Input and hidden dimension size.
        num_layers (int): Number of RNN layers.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 1,
    ):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(
            self,
            input: Tensor,
            hidden_state: Tensor | None = None,
    ):
        """
        Forward pass with optional hidden state propagation.

        Args:
            input (torch.Tensor): Latent representation.
            hidden_state (torch.Tensor, optional): Previous hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - S: Generated sequence shifted in time.
                - h_new: Updated hidden state.
        """
        s_outputs, h_new = self.rnn(input, hidden_state)
        s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        S = self.tanh(S)
        return S, h_new


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
        hidden_dim (int): Input and hidden dimension size.
        num_layers (int): Number of RNN layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
    ):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(
            self,
            input: Tensor,
            hidden_state: Tensor | None = None,
    ):
        """
        Forward pass with optional hidden state propagation.

        Args:
            input (torch.Tensor): Latent representation.
            hidden_state (torch.Tensor, optional): Previous hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Y_hat: Classification results.
                - h_new: Updated hidden state.
        """
        d_outputs, h_new = self.rnn(input, hidden_state)
        d_outputs = self.norm(d_outputs) # to change if discriminator becomes too strong
        Y_hat = self.fc(d_outputs)
        Y_hat = self.sigmoid(Y_hat)
        return Y_hat, h_new

class TimeGan(nn.Module):
    """
    Pure TimeGAN networks: Encoder, Recovery, Generator, Supervisor, Discriminator.

    This class wraps the submodules and provides forward methods that expose
    hidden state management for stateful generation.
    """

    def __init__(
        self,
        encoder_input_dim: int,
        hidden_dim: int,
        generator_input_dim: int,
        discriminator_input_dim: int,
        recovery_output_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.recovery = Recovery(
            output_dim=recovery_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.generator = Generator(
            input_dim=generator_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.supervisor = Supervisor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.discriminator = Discriminator(
            input_dim=discriminator_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def e_forward(self, x, hidden_state = None):
        return self.encoder(x, hidden_state)

    def r_forward(self, H, hidden_state = None):
        return self.recovery(H, hidden_state)

    def s_forward(self, H, hidden_state = None):
        return self.supervisor(H, hidden_state)

    def g_forward(self, z_input, hidden_state = None):
        return self.generator(z_input, hidden_state)

    def d_forward(self, H, hidden_state = None):
        return self.discriminator(H, hidden_state)
