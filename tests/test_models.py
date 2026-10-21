"""Model contracts: parameter matching at full size, gradient flow everywhere,
causality for dense arms, TBPTT state-handoff equivalence for recurrent arms."""
import pytest
import torch
import torch.nn.functional as F

from stage2 import config
from stage2.dense import build

# F is used by the quant test's hand-computed reference
assert F is not None

FULL_ARMS = ["T2", "R2", "GPTU", "GPTQ", "LSTMU"]


def test_full_size_params_matched():
    """phase0_spec: all arms 1.90-1.92M, matched +/-10%."""
    counts = {a: sum(p.numel() for p in build(a).parameters()) for a in FULL_ARMS}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo < 1.10, f"param mismatch beyond 10%: {counts}"
    for a, n in counts.items():
        assert 1.7e6 < n < 2.1e6, f"{a}={n}"


@pytest.mark.parametrize("arm", FULL_ARMS)
def test_gradient_reaches_every_parameter(tiny, arm):
    """KNOWN-BAD this catches: a mis-wired STE leaves the embedding with zero
    grad and the arm silently trains a fixed random code."""
    torch.manual_seed(0)
    m = build(arm)
    x = torch.randint(0, config.V, (2, 8))
    y = torch.randint(0, config.V, (2, 8))
    st = m.init_state(2, "cpu") if hasattr(m, "init_state") else None
    logits, _ = m(x, st)
    F.cross_entropy(logits.reshape(-1, config.V), y.reshape(-1)).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{arm}.{name} got no grad"
    emb = dict(m.named_parameters()).get("emb.weight")
    if emb is not None:
        assert emb.grad.abs().sum() > 0, f"{arm}: embedding grad is exactly zero"


@pytest.mark.parametrize("arm", ["GPTU", "GPTQ", "LSTMU"])
def test_causality_dense(tiny, arm):
    """Perturbing a future token must not change past logits. Catches a broken
    causal mask — the bug class that makes val BPC a fiction."""
    torch.manual_seed(0)
    m = build(arm).eval()
    x = torch.randint(0, config.V, (1, 16))
    x2 = x.clone()
    x2[0, 12] = (x2[0, 12] + 1) % config.V
    with torch.no_grad():
        st = m.init_state(1, "cpu") if hasattr(m, "init_state") else None
        a, _ = m(x, st)
        st = m.init_state(1, "cpu") if hasattr(m, "init_state") else None
        b, _ = m(x2, st)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5), f"{arm} leaks future"


@pytest.mark.parametrize("arm", ["T2", "R2", "LSTMU"])
def test_tbptt_chunk_equivalence(tiny, arm):
    """Full forward == chunked forward with state carry (grad-free). Catches
    dropped/reset state at chunk boundaries — which would silently turn the
    'context memory' claim into a 8-token model."""
    torch.manual_seed(0)
    m = build(arm).eval()
    x = torch.randint(0, config.V, (2, 16))
    with torch.no_grad():
        full, _ = m(x, m.init_state(2, "cpu"))
        st = m.init_state(2, "cpu")
        parts = []
        for lo in range(0, 16, 4):
            lg, st = m(x[:, lo:lo + 4], st)
            parts.append(lg)
        chunked = torch.cat(parts, dim=1)
    assert torch.allclose(full, chunked, atol=1e-4), f"{arm} state handoff broken"


def test_tbptt_equivalence_fails_on_state_reset(tiny):
    """KNOWN-BAD demonstration: resetting state between chunks must be DETECTED
    by the equivalence check above (i.e., produce different logits)."""
    torch.manual_seed(0)
    m = build("T2").eval()
    x = torch.randint(0, config.V, (2, 16))
    with torch.no_grad():
        full, _ = m(x, m.init_state(2, "cpu"))
        parts = []
        for lo in range(0, 16, 4):
            lg, _ = m(x[:, lo:lo + 4], m.init_state(2, "cpu"))  # reset = bug
            parts.append(lg)
        broken = torch.cat(parts, dim=1)
    assert not torch.allclose(full, broken, atol=1e-4), \
        "state reset was undetectable -- the equivalence test is vacuous"


def test_quant_actually_quantizes(tiny):
    """GPTQ must not silently be GPTU. The first version of this test passed
    for ANY matrix (review M2: '<=16 levels after dividing by amax/7' is a
    tautology). Real check: the forward OUTPUT must differ from the
    unquantized linear, and must equal the hand-computed fake-quant result."""
    torch.manual_seed(0)
    from stage2.dense import FakeQuantLinear
    lin = FakeQuantLinear(16, 16, bias=False)
    x = torch.randn(4, 16)
    lin.train()
    y = lin(x)          # first call also sets act_scale from this batch
    y2 = lin(x)
    dense = F.linear(x, lin.weight)
    assert not torch.allclose(y2, dense, atol=1e-5), \
        "quantized forward is identical to dense: GPTQ is silently GPTU"
    qmax = 7
    s_a = (lin.act_scale / qmax).clamp_min(1e-8)
    xq = (x / s_a).round().clamp(-qmax - 1, qmax) * s_a
    s_w = (lin.weight.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    wq = (lin.weight / s_w).round().clamp(-qmax - 1, qmax) * s_w
    assert torch.allclose(y2, F.linear(xq, wq), atol=1e-5)
