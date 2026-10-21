"""Training-loop contracts: every arm can memorize at tiny scale on CPU, deaths
are recorded not raised, eval matches first principles on a known model."""
import math

import numpy as np
import pytest
import torch

from stage2 import config, data
from stage2.dense import build
from stage2.train import eval_bpc, seed_everything, train_run

ARMS = ["T2", "R2", "GPTU", "GPTQ", "LSTMU"]


def _toy_batches(n_seq=8, seq=16):
    rng = np.random.default_rng(3)
    xs = rng.integers(0, config.V, size=(n_seq, seq + 1)).astype(np.int64)
    x = torch.from_numpy(xs[:, :-1])
    y = torch.from_numpy(xs[:, 1:])
    return lambda: (x, y)


@pytest.mark.parametrize("arm", ARMS)
def test_tiny_memorization_loss_drops(tiny, arm):
    """The Gate-0 mechanism in miniature: loss on a fixed batch must fall
    substantially. An arm that cannot overfit 8 sequences has a broken
    pipeline, and that must be visible BEFORE a pod is paid for."""
    seed_everything(0)
    m = build(arm)
    res = train_run(m, 400, _toy_batches(), "cpu", f"tiny_{arm}",
                    eval_every=100)
    assert res["death"] is None, res["death"]
    losses = [h["train_loss_bits"] for h in res["history"]]
    # calibrated against measured trajectories (GPT 0.02, LSTM 2.2, T2 3.8 bits
    # at step 400): require a >=1.2-bit drop from uniform and a falling curve.
    assert losses[-1] < math.log2(config.V) - 1.2, \
        f"{arm}: {losses}, no learning"
    assert losses[-1] < losses[0], f"{arm}: not decreasing: {losses}"


def test_nan_is_recorded_not_raised(tiny, monkeypatch):
    """KNOWN-BAD: a divergent run must come back as a death verdict."""
    monkeypatch.setattr(config, "LR_DENSE", 1e6)
    monkeypatch.setattr(config, "ARM_OVERRIDES", {})   # else GPTU takes 2e-3
    seed_everything(0)
    res = train_run(build("GPTU"), 30, _toy_batches(), "cpu", "boom",
                    eval_every=1000)
    assert res["death"] is not None and res["death"]["kind"] == "nan"


def test_eval_bpc_uniform_model_is_log2v(tiny):
    """A model with zeroed readout emits uniform logits -> BPC must be exactly
    log2(V). Catches nats/bits and averaging-denominator errors in one shot."""
    m = build("LSTMU")
    torch.nn.init.zeros_(m.readout.weight)
    torch.nn.init.zeros_(m.readout.bias)
    ids = np.arange(2000, dtype=np.uint8) % config.V
    r = eval_bpc(m, ids, "cpu")
    assert r["bpc"] == pytest.approx(math.log2(config.V), abs=1e-4)
    assert r["n_chars"] > 0


def test_eval_windows_do_not_overlap(tiny):
    """The eval definition is frozen: non-overlapping windows,
    every position scored once. n_chars must equal n_win * SEQ exactly."""
    m = build("LSTMU")
    S = config.SEQ
    ids = np.zeros(S * 5 + 3, dtype=np.uint8)
    r = eval_bpc(m, ids, "cpu")
    assert r["n_chars"] == 5 * S


def test_echo_diagnostic_fires_on_copy_model(tiny):
    """KNOWN-BAD analogue of stage1's echo regime: a model that copies its input
    must show p_argmax_curr ~ 1 and p_argmax_next low. Build one by hand."""

    class Copier(torch.nn.Module):
        def forward(self, x, st=None, counter=None):
            B, T = x.shape
            logits = torch.full((B, T, config.V), -10.0)
            logits.scatter_(-1, x.unsqueeze(-1), 10.0)
            return logits, None

    ids = (np.random.default_rng(0).integers(0, config.V, 4000)
           .astype(np.uint8))
    r = eval_bpc(Copier(), ids, "cpu")
    assert r["p_argmax_curr"] > 0.95
    assert r["p_argmax_next"] < 0.2


def test_seed_parity_across_arms(tiny, corpus_ids):
    """Two arms constructed with the same data seed must see identical batches
    (the cross-arm pairing claim is a code property, not hope)."""
    b1 = data.OrderedBatcher(corpus_ids, config.SEQ, config.BATCH, seed=2)
    b2 = data.OrderedBatcher(corpus_ids, config.SEQ, config.BATCH, seed=2)
    for _ in range(3):
        x1, _ = b1.next()
        x2, _ = b2.next()
        assert torch.equal(x1, x2)
