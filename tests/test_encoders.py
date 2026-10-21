"""Encoder contracts. T2's claim to be a LATENCY code and R2's claim to carry
nothing in timing are load-bearing for the whole study — so they are tested as
properties, not assumed from the code's intent."""
import torch

from stage2 import config
from stage2.spiking import count_train, latency_train


def test_latency_exactly_one_spike_binary_forward(tiny):
    e = torch.randn(8, 16)
    tr = latency_train(e)
    assert tr.shape == (8, config.W, 16)
    assert set(tr.detach().unique().tolist()) <= {0.0, 1.0}, "forward must be exactly binary"
    assert (tr.detach().sum(dim=1) == 1.0).all(), "exactly one spike per channel"


def test_latency_time_tracks_input(tiny):
    W = config.W
    lo = latency_train(torch.full((1, 4), -50.0)).detach()
    hi = latency_train(torch.full((1, 4), +50.0)).detach()
    t_lo = lo[0].argmax(dim=0)          # spike step per channel
    t_hi = hi[0].argmax(dim=0)
    assert (t_lo == 0).all(), "saturated-negative input -> earliest slot"
    assert (t_hi == W - 2).all(), "saturated-positive input -> latest slot"
    assert (t_hi > t_lo).all(), "spike time must increase with input"


def test_latency_gradient_flows(tiny):
    """Loss = expected spike TIME (grid-weighted), so gradient must flow through
    the timing itself. KNOWN-BAD baseline: with the STE term removed (hard
    train alone) this gradient is exactly zero — discrete scatter has no grad."""
    e = torch.randn(4, 8, requires_grad=True)
    grid = torch.arange(config.W, dtype=torch.float32).view(1, -1, 1)
    (latency_train(e) * grid).sum().backward()
    assert e.grad is not None and e.grad.abs().sum() > 0
    # KNOWN-BAD: rebuild the hard train WITHOUT the STE term; its time-weighted
    # loss must produce exactly zero gradient (discrete scatter has no grad),
    # proving the assertion above is doing real work.
    e2 = torch.randn(4, 8, requires_grad=True)
    W = config.W
    tc = (W - 2) * torch.sigmoid(e2)
    hard = torch.zeros(4, W, 8)
    hard.scatter_(1, tc.detach().floor().long().clamp(0, W - 1).unsqueeze(1), 1.0)
    loss = (hard * grid).sum()
    assert not loss.requires_grad, \
        "hard-only train should have NO gradient path -- STE test is vacuous"


def test_count_code_carries_nothing_in_timing(tiny):
    """Two inputs with the same COUNT must give the same pattern: timing is a
    fixed function of count, never of the analog value."""
    W = config.W
    e = torch.tensor([[10.0], [9.0]])          # both saturate -> count W
    tr = count_train(e).detach()
    assert (tr[0] == tr[1]).all()
    counts = (W * torch.sigmoid(e)).floor().clamp(0, W)
    assert (tr.sum(dim=1).squeeze(-1) == counts.squeeze(-1)).all()


def test_count_gradient_flows(tiny):
    e = torch.randn(4, 8, requires_grad=True)
    count_train(e).sum().backward()
    assert e.grad is not None and e.grad.abs().sum() > 0


def test_codes_differ(tiny):
    """The two encoders must actually produce different trains for the same
    input — otherwise the arm contrast is fictional."""
    e = torch.randn(6, 12)
    assert not torch.equal(latency_train(e).detach(), count_train(e).detach())
