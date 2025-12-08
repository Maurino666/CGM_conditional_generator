import torch
from torch import Tensor

from models import TimeGanModule

def test_timegan_autoencoder_phase_training_step_runs():
    """
    In autoencoder phase, TimeGanModule.training_step should return a float loss.
    """
    batch_size = 4
    seq_len = 10
    input_dim = 3

    model = TimeGanModule(
        input_dim=input_dim,
        hidden_dim=8,
        num_layers=1,
        noise_dim=4,
        lr=1e-3,
        g_steps_per_iter=1,  # keep it light for tests
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    model.set_phase(model.PHASE_AUTOENCODER)
    loss_value = model.training_step(x)

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_timegan_supervisor_phase_training_step_runs():
    """
    In supervisor phase, TimeGanModule.training_step should return a float loss.
    """
    batch_size = 3
    seq_len = 8
    input_dim = 2

    model = TimeGanModule(
        input_dim=input_dim,
        hidden_dim=6,
        num_layers=1,
        noise_dim=3,
        lr=1e-3,
        g_steps_per_iter=1,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    model.set_phase(model.PHASE_SUPERVISOR)
    loss_value = model.training_step(x)

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_timegan_adversarial_phase_training_step_returns_loss_dict():
    """
    In adversarial phase, TimeGanModule.training_step should return
    a dict with g_loss, er_loss, d_loss.
    """
    batch_size = 2
    seq_len = 6
    input_dim = 3

    model = TimeGanModule(
        input_dim=input_dim,
        hidden_dim=8,
        num_layers=1,
        noise_dim=4,
        lr=1e-3,
        g_steps_per_iter=1,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    model.set_phase(model.PHASE_ADVERSARIAL)
    losses = model.training_step(x)

    assert isinstance(losses, dict)
    for key in ("g_loss", "er_loss", "d_loss"):
        assert key in losses
        assert isinstance(losses[key], float)
        assert torch.isfinite(torch.tensor(losses[key]))


def test_timegan_generate_shape():
    """
    TimeGanModule.generate should produce sequences with shape
    (num_samples, seq_len, input_dim).
    """
    input_dim = 4
    model = TimeGanModule(
        input_dim=input_dim,
        hidden_dim=8,
        num_layers=1,
        noise_dim=4,
        lr=1e-3,
        g_steps_per_iter=1,
    )

    num_samples = 5
    seq_len = 7

    with torch.no_grad():
        samples = model.generate(num_samples=num_samples, seq_len=seq_len)

    assert isinstance(samples, Tensor)
    assert samples.shape == (num_samples, seq_len, input_dim)
