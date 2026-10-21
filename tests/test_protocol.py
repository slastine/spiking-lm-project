"""Protocol-scale additions: new arms, resume equivalence, buckets, probes.
Same house rule: guards must be shown to do real work."""
import math

import numpy as np
import pytest
import torch

from stage2 import config, data
from stage2.accounting import (gptfm_macs_per_token, lstm12_macs_per_token,
                               t2_training_macs_per_token,
                               transformer_macs_per_token)
from stage2.dense import build
from stage2.probes import (generate, make_induction_probes, novelty_8gram,
                           score_induction)
from stage2.train import eval_bpc, seed_everything, train_run


def test_lstm12_params_match_lstmu():
    a = sum(p.numel() for p in build("LSTMU").parameters())
    b = sum(p.numel() for p in build("LSTM12").parameters())
    assert a == b, "LSTM12 must be the identical body (unroll-only control)"


def test_lstm12_actually_unrolls(tiny):
    """Same weights, micro=12 vs plain: outputs must differ (the control is
    not a no-op) and shapes must match token count."""
    torch.manual_seed(0)
    m12 = build("LSTM12")
    m1 = build("LSTMU")
    m1.load_state_dict(m12.state_dict())
    x = torch.randint(0, config.V, (2, 8))
    with torch.no_grad():
        y12, _ = m12(x, m12.init_state(2, "cpu"))
        y1, _ = m1(x, m1.init_state(2, "cpu"))
    assert y12.shape == y1.shape == (2, 8, config.V)
    assert not torch.allclose(y12, y1), "12 micro-steps changed nothing"


def test_lstm12_chunk_equivalence(tiny):
    torch.manual_seed(0)
    m = build("LSTM12").eval()
    x = torch.randint(0, config.V, (2, 16))
    with torch.no_grad():
        full, _ = m(x, m.init_state(2, "cpu"))
        st = m.init_state(2, "cpu")
        parts = []
        for lo in range(0, 16, 4):
            lg, st = m(x[:, lo:lo + 4], st)
            parts.append(lg)
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_gptfm_size_and_compute_match():
    """Prereg §2: GPT-FM ≈19.4M params, MACs within ±10% of rho x GPT-µ."""
    n = sum(p.numel() for p in build("GPTFM").parameters())
    assert 18e6 < n < 21e6, n
    rho = t2_training_macs_per_token() / transformer_macs_per_token(config.SEQ)
    target = rho * transformer_macs_per_token(config.SEQ)
    got = gptfm_macs_per_token(config.SEQ)
    assert abs(got - target) / target < 0.10, (got, target, rho)
    assert 8.5 < rho < 10.5, f"rho drifted from the pinned 9.36: {rho}"


def test_lstm12_compute_matches_spiking():
    """The unroll control matches T2's training compute within 10%."""
    assert abs(lstm12_macs_per_token() - t2_training_macs_per_token()) \
        / t2_training_macs_per_token() < 0.10


def test_resume_bitwise_equivalence(tiny, tmp_path, monkeypatch):
    """Train 10 straight vs train-crash-resume at 5: final weights must match.
    KNOWN-BAD: skipping the batcher fast-forward must break the equivalence
    (proves the test detects divergence)."""
    monkeypatch.setattr(config, "CKPT_EVERY", 5)
    ids = np.random.default_rng(0).integers(0, config.V, 60_000).astype(np.uint8)

    def run(steps, resume=None, skip=0, snapshot_after=None):
        import shutil
        seed_everything(7)
        m = build("LSTMU")
        b = data.OrderedBatcher(ids, 16, 4, seed=3)
        if skip:
            b.skip(skip)
        calls = {"n": skip}

        def nb():
            # snapshot the ckpt file mid-run: this IS the crash scenario —
            # the checkpoint written at step 5 of the 10-step schedule
            if snapshot_after is not None and calls["n"] == snapshot_after:
                shutil.copy(tmp_path / "ck.pt", tmp_path / "ck5.pt")
            calls["n"] += 1
            return b.next()

        train_run(m, steps, nb, "cpu", "rz", eval_every=1000,
                  ckpt_path=str(tmp_path / "ck.pt"), resume=resume)
        return m

    ref = run(10, snapshot_after=5)
    ck5 = torch.load(tmp_path / "ck5.pt")
    assert ck5["step"] == 5, "snapshot did not capture the step-5 checkpoint"
    resumed = run(10, resume=ck5, skip=5)
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(),
                                  resumed.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"resume diverged at {n1}"
    # KNOWN-BAD: resume WITHOUT batcher fast-forward must NOT match
    bad = run(10, resume=ck5, skip=0)
    diff = any(not torch.allclose(p1, p2, atol=1e-6)
               for (_, p1), (_, p2) in zip(ref.named_parameters(),
                                           bad.named_parameters()))
    assert diff, "batcher misalignment was undetectable — test is vacuous"


def test_batcher_skip_exact(corpus_ids):
    a = data.OrderedBatcher(corpus_ids, 32, 4, seed=5)
    b = data.OrderedBatcher(corpus_ids, 32, 4, seed=5)
    for _ in range(3):
        a.next()
    b.skip(3)
    xa, ya = a.next()
    xb, yb = b.next()
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_bucket_bpc_sums_and_uniform():
    # full-scale SEQ (buckets are protocol semantics, not tiny-config ones)
    m = build("LSTMU")
    torch.nn.init.zeros_(m.readout.weight)
    torch.nn.init.zeros_(m.readout.bias)
    ids = np.arange(4000, dtype=np.uint8) % config.V
    r = eval_bpc(m, ids, "cpu")
    assert sum(r["bucket_n"]) == r["n_chars"] > 0
    for b in r["bucket_bpc"]:
        assert b == pytest.approx(math.log2(config.V), abs=1e-4)


def test_induction_probes_deterministic_and_sane(corpus_ids):
    x1, s1, sha1 = make_induction_probes(corpus_ids, n_per_dist=20)
    x2, _, sha2 = make_induction_probes(corpus_ids, n_per_dist=20)
    assert sha1 == sha2, "probe set must be a pure function of the frozen seed"
    assert x1.shape == (60, config.SEQ)
    assert (s1 < config.SEQ - 1).all() and (s1 >= 0).all()
    # each probe really contains the string twice
    for i in range(0, 60, 7):
        p2 = s1[i][0]
        w = x1[i]
        found = any((w[j:j + 6] == w[p2:p2 + 6]).all()
                    for j in range(0, p2 - 6))
        assert found, "second occurrence has no matching first occurrence"


def test_induction_scoring_runs(corpus_ids):
    # full-scale SEQ: probe distances {64,128,192} require 256-char windows
    torch.manual_seed(0)
    m = build("LSTMU")
    x, s, _ = make_induction_probes(corpus_ids, n_per_dist=4)
    r = score_induction(m, x, s, "cpu")
    for k in ("acc_d64", "acc_d128", "acc_d192"):
        assert 0.0 <= r[k] <= 1.0


def test_generation_deterministic_and_sized(tiny):
    torch.manual_seed(0)
    m = build("LSTMU")
    g1 = generate(m, [1, 2, 3], 40, "cpu", gen_seed=1)
    g2 = generate(m, [1, 2, 3], 40, "cpu", gen_seed=1)
    g3 = generate(m, [1, 2, 3], 40, "cpu", gen_seed=2)
    assert g1 == g2 and len(g1) == 40
    assert g1 != g3, "different gen seeds must differ"


def test_novelty_8gram_extremes():
    train = np.arange(1000, dtype=np.uint8) % 64
    copied = list(train[100:200])
    assert novelty_8gram(copied, train)["novel_frac"] == 0.0
    rng = np.random.default_rng(1)
    novel = list(rng.integers(0, 64, 100).astype(int))
    assert novelty_8gram(novel, train)["novel_frac"] > 0.9


def test_grid_saves_states_and_flags(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID", (4, 8))
    ids = np.random.default_rng(0).integers(0, config.V, 60_000).astype(np.uint8)
    seed_everything(1)
    m = build("LSTMU")
    b = data.OrderedBatcher(ids, 16, 4, seed=1)
    res = train_run(m, 10, b.next, "cpu", "gtest", eval_every=1000,
                    grid=config.GRID, grid_dir=str(tmp_path))
    assert (tmp_path / "grid_gtest_4.pt").exists()
    assert (tmp_path / "grid_gtest_8.pt").exists()
    grid_rows = [h for h in res["history"] if h.get("grid")]
    assert [h["step"] for h in grid_rows] == [4, 8]
    sd = torch.load(tmp_path / "grid_gtest_4.pt")
    assert "lstm.weight_ih_l0" in sd
