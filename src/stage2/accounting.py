"""Dual-counter efficiency accounting (the 64x rule).

RuntimeCounter: accumulates spikes from tensors handed over by the model's
forward loop (add_step), fan-outs read off the model's weight shapes.

HookRecorder: a genuinely independent second path — nn.Module forward
pre-hooks capture the spike tensors as they enter w_rec1/w_rec2/w_in, and
tokens are counted from the input ids' shape. It never touches add_step.
The first draft's "static check" consumed rates derived from the runtime
counter's own accumulators — algebraically circular, certified-correct-by-
construction (adversarial review finding 1; stage1 defect-21 pattern). This
version measures twice or it doesn't count.

Units: ALL cross-arm op counts are MACs (1 synaptic accumulate = 1 MAC-
equivalent by declaration; exchange-rate sensitivity is a reporting concern,
proposal §5 C3). No factor of 2 anywhere — the first draft mixed 2x-FLOP and
1x-MAC conventions across arms (review finding 2), which would have silently
flattered the spiking arms in the headline ratio.

Every reported quantity is {"value": v, "per": unit}; combine() hard-fails on
mismatched units instead of silently converting.
"""
import torch

from . import config

UNITS = ("token", "sequence", "batch", "step", "run", "neuron_step")


def qty(value, per):
    assert per in UNITS
    return {"value": float(value), "per": per}


def combine(a, b, fn):
    """Any arithmetic across quantities must go through here."""
    if a["per"] != b["per"]:
        raise ValueError(f"unit mismatch: {a['per']} vs {b['per']}")
    return {"value": fn(a["value"], b["value"]), "per": a["per"]}


def _fanouts(model):
    return {
        "in": model.w_in.weight.shape[0],                          # ch -> H
        "s1": model.w_rec1.weight.shape[0] + model.w_ff.weight.shape[0],
        "s2": model.w_rec2.weight.shape[0] + 1,                    # +1 lp acc
        "H": model.w_rec1.weight.shape[0],
    }


def _dense_macs_per_token(H):
    """Charged in full: the recurrent tick is not free.
    leak/reset for v1, v2 AND the lp2 low-pass (3 per-neuron mults/step),
    readout 2H*V, encoder ~D_EMB."""
    return 3 * H * config.W + 2 * H * config.V + config.D_EMB


def _synops(in_spk, s1_spk, s2_spk, fo):
    return in_spk * fo["in"] + s1_spk * fo["s1"] + s2_spk * fo["s2"]


class RuntimeCounter:
    """Path A: counts events handed to add_step by the model's forward loop."""

    def __init__(self, model):
        self.fo = _fanouts(model)
        self.z()

    def z(self):
        self.in_spk = self.s1_spk = self.s2_spk = 0.0
        self.steps = 0
        self.tokens = 0
        self._dev = None

    def add_step(self, in_spikes, s1, s2):
        with torch.no_grad():
            if self._dev is None:
                self._dev = torch.zeros(3, device=s1.device)
            self._dev[0] += in_spikes.sum()
            self._dev[1] += s1.sum()
            self._dev[2] += s2.sum()
            self.steps += 1

    def add_tokens(self, n):
        self.tokens += int(n)

    def _sync(self):
        if self._dev is not None:
            v = self._dev.tolist()
            self.in_spk += v[0]
            self.s1_spk += v[1]
            self.s2_spk += v[2]
            self._dev = None

    def report(self):
        self._sync()
        if self.tokens == 0:
            raise ValueError("no tokens counted")
        tk = self.tokens
        H = self.fo["H"]
        return {
            "spikes_in": qty(self.in_spk / tk, "token"),
            "spikes_l1": qty(self.s1_spk / tk, "token"),
            "spikes_l2": qty(self.s2_spk / tk, "token"),
            "synops": qty(_synops(self.in_spk, self.s1_spk, self.s2_spk,
                                  self.fo) / tk, "token"),
            "dense_macs": qty(_dense_macs_per_token(H), "token"),
            "rate_l1": qty(self.s1_spk / (tk * config.W * H), "neuron_step"),
            "rate_l2": qty(self.s2_spk / (tk * config.W * H), "neuron_step"),
            "tokens": qty(tk, "run"),
        }


class HookRecorder:
    """Path B: independent measurement via forward pre-hooks. Shares no
    accumulator, no call site, and no token counter with RuntimeCounter."""

    def __init__(self, model):
        self.fo = _fanouts(model)
        self.s1 = self.s2 = self.inp = 0.0
        self.tokens = 0
        self._handles = [
            model.w_rec1.register_forward_pre_hook(self._h("s1")),
            model.w_rec2.register_forward_pre_hook(self._h("s2")),
            model.w_in.register_forward_pre_hook(self._h("inp")),
            model.register_forward_pre_hook(self._tokens),
        ]

    def _h(self, name):
        def hook(mod, args):
            with torch.no_grad():
                setattr(self, name, getattr(self, name) + float(args[0].sum()))
        return hook

    def _tokens(self, mod, args):
        x = args[0]
        self.tokens += int(x.shape[0] * x.shape[1])

    def detach(self):
        for h in self._handles:
            h.remove()

    def synops_per_token(self):
        if self.tokens == 0:
            raise ValueError("no tokens seen by hooks")
        return _synops(self.inp, self.s1, self.s2, self.fo) / self.tokens


def transformer_macs_per_token(ctx):
    """Pure MACs. Weight MACs: qkv 3d^2 + proj d^2 + mlp 8d^2 = 12d^2 per
    layer. Attention at context ctx: QK^T (d*ctx) + AV (ctx*d) per layer.
    Tied unembed d*V. KV-cache reads are BYTES, reported separately and
    declared incommensurable (proposal §7) — never folded in here."""
    d, L, V = config.GPT_D, config.GPT_LAYERS, config.V
    return L * 12 * d * d + L * 2 * d * ctx + d * V


def lstm_macs_per_token():
    """Pure MACs, same convention."""
    h, e, L, V = config.LSTM_H, config.D_EMB, config.LSTM_LAYERS, config.V
    return 4 * h * (e + h) + (L - 1) * 4 * h * (2 * h) + h * V


def lstm12_macs_per_token():
    """Unroll control: LSTM body cost x micro-steps, readout once per token."""
    h, e, L, V = config.LSTM_H, config.D_EMB, config.LSTM_LAYERS, config.V
    body = 4 * h * (e + h) + (L - 1) * 4 * h * (2 * h)
    return body * config.LSTM12_MICRO + h * V


def gptfm_macs_per_token(ctx):
    """Compute-matched transformer, same pure-MAC convention."""
    d, L, V = config.GPTFM_D, config.GPTFM_LAYERS, config.V
    return L * 12 * d * d + L * 2 * d * ctx + d * V


def t2_training_macs_per_token():
    """T2's dense-training charge (rho convention: surrogate BPTT
    gets no sparsity credit — input, recurrent, ff, readout, leak all dense)."""
    H, C, W, V = config.N_HID, config.D_EMB, config.W, config.V
    per_step = (C + 1) * H + H * H * 3 + 3 * H      # in + rec1/ff/rec2 + leaks
    return per_step * W + 2 * H * V + C


def agreement(a, b, tol=0.05):
    if b == 0:
        return a == 0
    return abs(a - b) / b <= tol
