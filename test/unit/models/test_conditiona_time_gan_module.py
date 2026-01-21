import torch
from torch import Tensor

from models import ConditionalTimeGanModule

def _make_conditional_timegan(cond_dim: int) -> ConditionalTimeGanModule:
    return ConditionalTimeGanModule(
        cond_dim=cond_dim,
        hidden_dim=8,
        num_layers=1,
        noise_dim=4,
        lr=1e-3,
        g_steps_per_iter=1,
    )


def test_conditional_timegan_autoencoder_phase_training_step_runs():
    """
    In autoencoder phase, ConditionalTimeGanModule.training_step should
    accept (y, c) and return a float.
    """
    batch_size = 4
    seq_len = 10
    cond_dim = 3

    model = _make_conditional_timegan(cond_dim=cond_dim)

    y = torch.randn(batch_size, seq_len, 1)
    c = torch.randn(batch_size, seq_len, cond_dim)

    model.set_phase(model.PHASE_AUTOENCODER)
    loss_value = model.training_step((y, c))

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_conditional_timegan_supervisor_phase_training_step_runs():
    """
    In supervisor phase, ConditionalTimeGanModule.training_step should
    accept (y, c) and return a float.
    """
    batch_size = 3
    seq_len = 8
    cond_dim = 2

    model = _make_conditional_timegan(cond_dim=cond_dim)

    y = torch.randn(batch_size, seq_len, 1)
    c = torch.randn(batch_size, seq_len, cond_dim)

    model.set_phase(model.PHASE_SUPERVISOR)
    loss_value = model.training_step((y, c))

    assert isinstance(loss_value, float)
    assert loss_value >= 0.0 or torch.isfinite(torch.tensor(loss_value))


def test_conditional_timegan_adversarial_phase_training_step_returns_loss_dict():
    """
    In adversarial phase, ConditionalTimeGanModule.training_step should
    accept (y, c) and return a dict of losses.
    """
    batch_size = 2
    seq_len = 6
    cond_dim = 3

    model = _make_conditional_timegan(cond_dim=cond_dim)

    y = torch.randn(batch_size, seq_len, 1)
    c = torch.randn(batch_size, seq_len, cond_dim)

    model.set_phase(model.PHASE_ADVERSARIAL)
    losses = model.training_step((y, c))

    assert isinstance(losses, dict)
    for key in ("g_loss", "er_loss", "d_loss"):
        assert key in losses
        assert isinstance(losses[key], float)
        assert torch.isfinite(torch.tensor(losses[key]))


def test_conditional_timegan_generate_shape():
    """
    ConditionalTimeGanModule.generate should produce target sequences
    of shape (batch_size, seq_len, 1) given a conditional sequence.
    """
    cond_dim = 4
    model = _make_conditional_timegan(cond_dim=cond_dim)

    batch_size = 5
    seq_len = 7
    cond_seq = torch.randn(batch_size, seq_len, cond_dim)

    with torch.no_grad():
        y_hat = model.generate(cond_seq)

    assert isinstance(y_hat, Tensor)
    assert y_hat.shape == (batch_size, seq_len, 1)
