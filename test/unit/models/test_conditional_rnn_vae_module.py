import torch
from torch import Tensor

from models import ConditionalRnnVaeModule  # adjust import


def test_conditional_rnn_vae_training_step_runs_and_returns_float():
    """
    ConditionalRnnVaeModule.training_step should run on (y, c) batch and
    return a Python float loss.
    """
    batch_size = 4
    seq_len = 10
    cond_dim = 3

    model = ConditionalRnnVaeModule(
        cond_dim=cond_dim,
        hidden_dim=8,
        latent_dim=4,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    y = torch.randn(batch_size, seq_len, 1)
    c = torch.randn(batch_size, seq_len, cond_dim)

    loss_value = model.training_step((y, c))

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_conditional_rnn_vae_validation_step_returns_scalar_tensor():
    """
    ConditionalRnnVaeModule.validation_step should return a scalar loss tensor.
    """
    batch_size = 3
    seq_len = 8
    cond_dim = 2

    model = ConditionalRnnVaeModule(
        cond_dim=cond_dim,
        hidden_dim=6,
        latent_dim=3,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    y = torch.randn(batch_size, seq_len, 1)
    c = torch.randn(batch_size, seq_len, cond_dim)

    with torch.no_grad():
        loss = model.validation_step((y, c))

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0  # scalar tensor


def test_conditional_rnn_vae_sample_shape():
    """
    ConditionalRnnVaeModule.sample should generate sequences with shape
    (num_samples, seq_len, 1) in target space.
    """
    cond_dim = 4
    model = ConditionalRnnVaeModule(
        cond_dim=cond_dim,
        hidden_dim=8,
        latent_dim=4,
        num_layers=1,
        rnn_type="gru",
        beta=1.0,
        lr=1e-3,
        grad_clip=1.0,
    )

    num_samples = 5
    seq_len = 7

    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, seq_len=seq_len)

    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, seq_len, 1)
