"""Dense arms: GPT-u (nanoGPT-style), GPT-u-Q (W4A4 fake-quant QAT), LSTM-u."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class FakeQuantLinear(nn.Linear):
    """W4A4 QAT: per-out-channel symmetric 4-bit weights, per-tensor EMA activation
    scale, straight-through gradients. Forward is fake-quant (fp compute on
    quantized values) — the standard QAT contract."""

    BITS = 4

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.register_buffer("act_scale", torch.tensor(0.0))

    def forward(self, x):
        qmax = 2 ** (self.BITS - 1) - 1
        if self.training:
            with torch.no_grad():
                m = x.abs().amax()
                if float(self.act_scale) == 0.0:
                    self.act_scale.copy_(m)      # cold-start: no early clipping
                else:
                    self.act_scale.mul_(0.9).add_(0.1 * m)
        s_a = (self.act_scale / qmax).clamp_min(1e-8)
        xq = x + (((x / s_a).round().clamp(-qmax - 1, qmax) * s_a) - x).detach()
        s_w = (self.weight.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
        w = self.weight
        wq = w + (((w / s_w).round().clamp(-qmax - 1, qmax) * s_w) - w).detach()
        return F.linear(xq, wq, self.bias)


class Block(nn.Module):
    def __init__(self, lin, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = lin(d, 3 * d, bias=False)
        self.proj = lin(d, d, bias=False)
        self.mlp_up = lin(d, 4 * d, bias=False)
        self.mlp_dn = lin(4 * d, d, bias=False)
        self.h = h

    def forward(self, x):
        B, T, d = x.shape
        p = getattr(self, "pdrop", config.GPT_DROPOUT) if self.training else 0.0
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        shp = (B, T, self.h, d // self.h)
        q, k, v = (t.view(shp).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, d)), p,
                          self.training)
        return x + F.dropout(self.mlp_dn(F.gelu(self.mlp_up(self.ln2(x)))), p,
                             self.training)


class GPTMu(nn.Module):
    """arm in {"GPTU", "GPTQ", "GPTFM"}. Tied embeddings; ctx = config.SEQ.
    GPTFM is the compute-matched transformer (prereg §2)."""

    def __init__(self, arm: str):
        super().__init__()
        assert arm in ("GPTU", "GPTQ", "GPTFM")
        self.arm = arm
        lin = FakeQuantLinear if arm == "GPTQ" else nn.Linear
        if arm == "GPTFM":
            L, h, d = config.GPTFM_LAYERS, config.GPTFM_HEADS, config.GPTFM_D
        else:
            L, h, d = config.GPT_LAYERS, config.GPT_HEADS, config.GPT_D
        self.n_layers = L
        self.pdrop = config.ARM_OVERRIDES.get(arm, {}).get(
            "dropout", config.GPT_DROPOUT)
        self.emb = nn.Embedding(config.V, d)
        self.pos = nn.Embedding(config.SEQ, d)
        self.blocks = nn.ModuleList(Block(lin, d, h) for _ in range(L))
        for b in self.blocks:
            b.pdrop = self.pdrop
        self.ln_f = nn.LayerNorm(d)
        # nanoGPT reference init, matched EXACTLY (review: an all-linears
        # sqrt-scale made every GPT weight 3.46x small — a depressed baseline
        # would manufacture a false "spiking competitive" verdict): std 0.02
        # everywhere, residual-out projections scaled by 1/sqrt(2*n_layer).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            for w in (b.proj.weight, b.mlp_dn.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * self.n_layers))
        # tied unembed makes the embedding the output matrix: N(0,1) default
        # init produces ~sqrt(d)-scale logits and a diverging first-30-steps
        # (caught by the CPU smoke). nanoGPT-standard 0.02 for emb and pos.
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, st=None, counter=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        h = F.dropout(h, self.pdrop if self.training else 0.0, self.training)
        for b in self.blocks:
            h = b(h)
        return F.linear(self.ln_f(h), self.emb.weight), None   # tied unembed


class LSTMMu(nn.Module):
    def __init__(self, arm: str = "LSTMU"):
        super().__init__()
        self.arm = arm
        self.emb = nn.Embedding(config.V, config.D_EMB)
        self.lstm = nn.LSTM(config.D_EMB, config.LSTM_H, config.LSTM_LAYERS,
                            batch_first=True)
        self.readout = nn.Linear(config.LSTM_H, config.V)

    def init_state(self, B, device):
        z = torch.zeros(config.LSTM_LAYERS, B, config.LSTM_H, device=device)
        return (z, z.clone())

    def detach_state(self, st):
        return (st[0].detach(), st[1].detach())

    def forward(self, x, st=None, counter=None):
        h, st = self.lstm(self.emb(x), st)
        return self.readout(h), st


class LSTM12(LSTMMu):
    """Unroll-depth control (prereg §2): identical LSTM-µ body, each token's
    embedding presented for 12 consecutive micro-steps, readout at the 12th.
    Matches T2/R2 in params, serial depth, and training compute at once — the
    deflation control for 'a recurrent SNN beats an LSTM' (prereg S2)."""

    def __init__(self):
        super().__init__(arm="LSTM12")

    def forward(self, x, st=None, counter=None):
        B, T = x.shape
        M = config.LSTM12_MICRO
        e = self.emb(x)                               # [B, T, C]
        e = e.unsqueeze(2).expand(B, T, M, -1).reshape(B, T * M, -1)
        h, st = self.lstm(e, st)
        h = h[:, M - 1::M]                            # 12th micro-step per token
        return self.readout(h), st


def build(arm: str) -> nn.Module:
    from .spiking import SpikingLM
    return {"T2": lambda: SpikingLM("T2"), "R2": lambda: SpikingLM("R2"),
            "GPTU": lambda: GPTMu("GPTU"), "GPTQ": lambda: GPTMu("GPTQ"),
            "GPTFM": lambda: GPTMu("GPTFM"), "LSTMU": lambda: LSTMMu(),
            "LSTM12": lambda: LSTM12()}[arm]()
