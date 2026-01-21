import torch
from torch import Tensor

from models import RnnVaeModule  # adjust import path


def test_rnn_vae_module_training_step_runs_and_returns_float():
    """
    RnnVaeModule.training_step should run on a small random batch and
    return a Python float loss.
    """
    batch_size = 4
    seq_len = 10
    input_dim = 3

    model = RnnVaeModule(
        input_dim=input_dim,
        hidden_dim=8,
        latent_dim=4,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    loss_value = model.training_step(x)

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_rnn_vae_module_validation_step_returns_scalar_tensor():
    """
    RnnVaeModule.validation_step should return a scalar loss tensor.
    """
    batch_size = 3
    seq_len = 8
    input_dim = 2

    model = RnnVaeModule(
        input_dim=input_dim,
        hidden_dim=6,
        latent_dim=3,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        loss = model.validation_step(x)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0  # scalar tensor


def test_rnn_vae_module_sample_shape():
    """
    RnnVaeModule.sample should generate sequences with the expected shape.
    """
    input_dim = 5
    model = RnnVaeModule(
        input_dim=input_dim,
        hidden_dim=8,
        latent_dim=4,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    num_samples = 7
    seq_len = 12

    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, seq_len=seq_len)

    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, seq_len, input_dim)
