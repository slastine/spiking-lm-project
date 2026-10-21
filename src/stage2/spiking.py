"""T2 (latency code) and R2 (count code) on a shared 2-layer recurrent LIF core.

Context memory lives in the carried membrane/spike state across token windows;
TBPTT detaches it at chunk boundaries. The two arms differ ONLY in the encoder:
T2 places one spike per channel at an input-dependent time; R2 emits an
input-dependent COUNT of evenly spaced spikes (no information in timing).

Discrete spike placement kills gradients, so both encoders use a straight-through
estimator: forward is the hard binary train, backward flows through a soft
surrogate. If the STE is wired wrong the embedding receives zero gradient — the
gradient-flow test exists to catch exactly that.
"""
import torch
import torch.nn as nn

from . import config


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, slope):
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v >= 0.0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        # fast-sigmoid surrogate (SuperSpike); slope is per-model so tuning
        # winners apply to T2 without silently retuning R2 (prereg §2)
        return g / (1.0 + ctx.slope * v.abs()) ** 2, None


def spike(v, slope=None):
    return _SpikeFn.apply(v, slope if slope is not None
                          else config.SURROGATE_SLOPE)


def latency_train(e: torch.Tensor) -> torch.Tensor:
    """T2 encoder. e: [B, C] analog values -> [B, W, C] binary, exactly one spike
    per channel, at step floor((W-2)*sigmoid(e)). STE backward via a softmax bump
    over the window centred on the continuous time."""
    B, C = e.shape
    W = config.W
    a = torch.sigmoid(e)
    tc = (W - 2) * a                                    # continuous time in [0, W-2]
    grid = torch.arange(W, device=e.device, dtype=e.dtype).view(1, W, 1)
    soft = torch.softmax(-(grid - tc.unsqueeze(1)) ** 2 / 0.5, dim=1)
    hard = torch.zeros(B, W, C, device=e.device, dtype=e.dtype)
    hard.scatter_(1, tc.detach().floor().long().clamp(0, W - 1).unsqueeze(1), 1.0)
    # parenthesized so the STE term is EXACTLY zero in forward (left-assoc
    # (hard+soft)-soft rounds and ships non-binary spikes -- caught by test)
    return hard + (soft - soft.detach())


def count_train(e: torch.Tensor) -> torch.Tensor:
    """R2 encoder. k = floor(W*sigmoid(e)) spikes, Bresenham-spaced. The spacing is
    a fixed function of the count, so timing carries nothing beyond rate. STE
    backward via the uniform-rate soft train."""
    B, C = e.shape
    W = config.W
    a = torch.sigmoid(e)
    k = (W * a).detach().floor().clamp(0, W)            # [B, C] spike counts
    idx = torch.arange(W, device=e.device, dtype=e.dtype).view(1, W, 1)
    # Bresenham: spike at step s iff floor((s+1)*k/W) > floor(s*k/W)
    hard = (((idx + 1) * k.unsqueeze(1) / W).floor()
            - (idx * k.unsqueeze(1) / W).floor()).clamp(0, 1)
    # per-step expected spike is a (window total ~= k), NOT a/W: a 1/W-scaled
    # surrogate attenuates R2's embedding gradient by W relative to T2's,
    # an uncontrolled second variable inside the arm contrast (review M4)
    soft = a.unsqueeze(1).expand(B, W, C)
    return hard + (soft - soft.detach())


class SpikingLM(nn.Module):
    """arm in {"T2", "R2"}; vocab-in embedding, LIF x2, linear readout."""

    def __init__(self, arm: str):
        super().__init__()
        assert arm in ("T2", "R2")
        self.arm = arm
        H, C = config.N_HID, config.D_EMB
        self.emb = nn.Embedding(config.V, C)
        self.w_in = nn.Linear(C + 1, H, bias=False)      # +1 clock channel
        self.w_rec1 = nn.Linear(H, H, bias=False)
        self.w_ff = nn.Linear(H, H, bias=True)
        self.w_rec2 = nn.Linear(H, H, bias=False)
        self.readout = nn.Linear(2 * H, config.V, bias=True)
        self.slope = config.ARM_OVERRIDES.get(arm, {}).get(
            "slope", config.SURROGATE_SLOPE)
        for w in (self.w_rec1.weight, self.w_rec2.weight):
            nn.init.orthogonal_(w, gain=0.5)

    def init_state(self, B, device):
        H = config.N_HID
        z = lambda: torch.zeros(B, H, device=device)
        return {"v1": z(), "s1": z(), "v2": z(), "s2": z()}

    def detach_state(self, st):
        return {k: v.detach() for k, v in st.items()}

    def forward(self, x, st, counter=None):
        """x: [B, T] token ids. Returns logits [B, T, V], new state.

        counter, if given, receives per-layer spike totals (accounting hooks)."""
        B, T = x.shape
        W, H = config.W, config.N_HID
        e = self.emb(x)                                  # [B, T, C]
        enc = latency_train if self.arm == "T2" else count_train
        trains = enc(e.reshape(B * T, -1)).reshape(B, T, W, -1)
        clock = torch.zeros(B, T, W, 1, device=x.device, dtype=trains.dtype)
        clock[:, :, 0, 0] = 1.0
        trains = torch.cat([trains, clock], dim=-1)      # [B, T, W, C+1]
        # input currents for all steps at once (dense op, charged in accounting)
        cur_in = self.w_in(trains)                       # [B, T, W, H]
        v1, s1, v2, s2 = st["v1"], st["s1"], st["v2"], st["s2"]
        logits = []
        for t in range(T):
            lp2 = torch.zeros(B, H, device=x.device)
            for w in range(W):
                v1 = config.ALPHA * v1 + cur_in[:, t, w] + self.w_rec1(s1) \
                    - config.V_TH * s1
                s1 = spike(v1 - config.V_TH, self.slope)
                v2 = config.ALPHA * v2 + self.w_ff(s1) + self.w_rec2(s2) \
                    - config.V_TH * s2
                s2 = spike(v2 - config.V_TH, self.slope)
                lp2 = config.LP_ALPHA * lp2 + s2
                if counter is not None:
                    counter.add_step(trains[:, t, w], s1, s2)
            logits.append(self.readout(torch.cat([lp2, v2], dim=-1)))
        if counter is not None:
            counter.add_tokens(B * T)
        st = {"v1": v1, "s1": s1, "v2": v2, "s2": s2}
        return torch.stack(logits, dim=1), st
