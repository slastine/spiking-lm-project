"""The 64x rule, executable. Two genuinely independent measurement paths
(RuntimeCounter via add_step; HookRecorder via module pre-hooks) must agree —
and the corruption tests prove disagreement CAN fire through the production
pipeline, not just on literals (both reviews, finding 1)."""
import pytest
import torch

from stage2 import config
from stage2.accounting import (HookRecorder, RuntimeCounter, agreement,
                               combine, lstm_macs_per_token, qty,
                               transformer_macs_per_token)
from stage2.dense import build
from stage2.train import accounting_pass


def test_unit_mismatch_hard_fails():
    with pytest.raises(ValueError, match="unit mismatch"):
        combine(qty(1, "token"), qty(1, "batch"), lambda a, b: a + b)


def test_qty_rejects_unknown_unit():
    with pytest.raises(AssertionError):
        qty(1, "trial")     # stage1's 64x was a per-batch/per-trial confusion


def test_runtime_counter_hand_fixture(tiny, monkeypatch):
    """Hand arithmetic (N_HID=3, D_EMB=2 -> w_in has 3 input cols incl clock):
    fan_in = 3, fan_s1 = 3+3 = 6, fan_s2 = 3+1 = 4.
    Feed 2 steps, 1 token: in totals 2, s1 totals 3, s2 totals 3.
    SynOps = 2*3 + 3*6 + 3*4 = 36 per token."""
    monkeypatch.setattr(config, "N_HID", 3)
    monkeypatch.setattr(config, "D_EMB", 2)
    monkeypatch.setattr(config, "W", 2)
    m = build("T2")
    c = RuntimeCounter(m)
    c.add_step(torch.tensor([[1., 0., 1.]]), torch.tensor([[1., 0., 0.]]),
               torch.tensor([[0., 1., 1.]]))
    c.add_step(torch.tensor([[0., 0., 0.]]), torch.tensor([[1., 1., 0.]]),
               torch.tensor([[0., 0., 1.]]))
    c.add_tokens(1)
    r = c.report()
    assert r["synops"]["value"] == 36.0
    assert r["synops"]["per"] == "token"
    assert r["rate_l1"]["per"] == "neuron_step"


def test_hook_vs_runtime_agree_on_real_model(tiny):
    """The two independent paths must agree exactly on a clean forward."""
    torch.manual_seed(0)
    m = build("T2").eval()
    c = RuntimeCounter(m)
    h = HookRecorder(m)
    x = torch.randint(0, config.V, (2, 6))
    with torch.no_grad():
        m(x, m.init_state(2, "cpu"), c)
    h.detach()
    r = c.report()
    assert h.tokens == r["tokens"]["value"] == 12
    assert h.synops_per_token() == pytest.approx(r["synops"]["value"], rel=1e-9)
    # T2 property: input spikes/token == D_EMB latency spikes + 1 clock spike
    assert r["spikes_in"]["value"] == config.D_EMB + 1


def test_accounting_pass_catches_runtime_corruption(tiny, monkeypatch):
    """KNOWN-BAD through the PRODUCTION path: a counter that drops half its
    add_step calls (a real bug class: counting per-chunk instead of per-step)
    must flip efficiency_status to EMBARGOED."""
    torch.manual_seed(0)
    m = build("T2").eval()
    ids = (torch.randint(0, config.V, (3000,)).numpy().astype("uint8"))
    orig = RuntimeCounter.add_step
    state = {"n": 0}

    def broken(self, in_spikes, s1, s2):
        state["n"] += 1
        if state["n"] % 2 == 0:
            return              # silently drop half the events
        orig(self, in_spikes, s1, s2)

    monkeypatch.setattr(RuntimeCounter, "add_step", broken)
    rep = accounting_pass(m, ids, "cpu")
    assert rep["efficiency_status"] == "EMBARGOED", \
        "a 2x undercount sailed through the agreement check"


def test_accounting_pass_clean_is_ok(tiny):
    torch.manual_seed(0)
    m = build("T2").eval()
    ids = (torch.randint(0, config.V, (3000,)).numpy().astype("uint8"))
    rep = accounting_pass(m, ids, "cpu")
    assert rep["efficiency_status"] == "OK"
    assert rep["counters_agree_5pct"] is True


def test_agreement_tolerance_band():
    assert not agreement(36.0 * 64, 36.0)    # the stage1 bug magnitude
    assert not agreement(36.0 * 1.1, 36.0)   # and a subtle 10%
    assert agreement(36.0 * 1.04, 36.0)      # inside the 5% band


def test_transformer_macs_pure_mac_convention(tiny):
    """Independent shape walk in PURE MACs — no factor of 2 anywhere. The
    first draft mixed 2x-FLOP weight terms with 1x-MAC attention (reviews:
    silently flatters spiking arms); this test hand-sums from layer shapes."""
    d, L, V = config.GPT_D, config.GPT_LAYERS, config.V
    ctx = 10
    per_layer = 0
    per_layer += d * (3 * d)        # qkv matmul, one token row
    per_layer += d * d              # attn out proj
    per_layer += d * (4 * d)        # mlp up
    per_layer += (4 * d) * d        # mlp down
    per_layer += d * ctx            # QK^T against ctx cached keys
    per_layer += ctx * d            # AV against ctx cached values
    total = L * per_layer + d * V   # + tied unembed
    assert transformer_macs_per_token(ctx) == total


def test_lstm_macs_pure_mac_convention(tiny):
    h, e, L, V = config.LSTM_H, config.D_EMB, config.LSTM_LAYERS, config.V
    manual = 4 * h * (e + h) + (L - 1) * 4 * h * (h + h) + h * V
    assert lstm_macs_per_token() == manual


def test_counter_refuses_zero_tokens(tiny):
    with pytest.raises(ValueError, match="no tokens"):
        RuntimeCounter(build("T2")).report()
