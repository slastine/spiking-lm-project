"""Unified train/eval for all five arms. One loop, one scoring function — arms
must not diverge in anything but the model (stage1's R4 lesson: separate
pipelines mean separate failure modes needing separate controls).

Token parity: every arm takes one optimizer step per batch of BATCH x SEQ tokens.
Recurrent arms process the batch in TBPTT chunks (state detached at chunk
boundaries, loss backward per chunk to bound the graph, gradients accumulate)
so step counts and token counts are identical across arms.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config, data
from .accounting import HookRecorder, RuntimeCounter, agreement, qty


def is_recurrent(model):
    return hasattr(model, "init_state")


def lr_at(step, total, peak):
    # warmup scales down for short runs: a fixed 200-step warmup means a
    # 120-step smoke never reaches 60% of peak and "no learning" conclusions
    # from short runs are schedule artifacts (review M5)
    warm = min(config.WARMUP, max(1, total // 10))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * min(t, 1.0))))


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)


def batch_loss(model, x, y, st=None, counter=None, chunk=None):
    """Forward+loss. For recurrent arms, iterates TBPTT chunks and calls
    backward per chunk when training (grad enabled); returns mean loss value,
    final state. For dense arms, single forward; caller does backward."""
    if not is_recurrent(model):
        logits, _ = model(x, None, counter)
        loss = F.cross_entropy(logits.reshape(-1, config.V), y.reshape(-1))
        if torch.is_grad_enabled():
            loss.backward()
        return float(loss.detach()), None
    chunk = chunk or config.TBPTT
    B, T = x.shape
    st = st or model.init_state(B, x.device)
    total = 0.0
    for lo in range(0, T, chunk):
        xc, yc = x[:, lo:lo + chunk], y[:, lo:lo + chunk]
        logits, st = model(xc, st, counter)
        loss = F.cross_entropy(logits.reshape(-1, config.V), yc.reshape(-1))
        if torch.is_grad_enabled():
            (loss * xc.shape[1] / T).backward()
        st = model.detach_state(st)
        total += float(loss.detach()) * xc.shape[1] / T
    return total, st


@torch.no_grad()
def eval_bpc(model, eval_ids, device, max_windows=None, counter=None):
    """Non-overlapping SEQ-char windows, all positions scored, state reset per
    window (frozen eval definition — identical for every arm). Returns BPC and
    the echo diagnostic."""
    model.eval()
    S = config.SEQ
    n_win = (len(eval_ids) - 1) // S
    if max_windows:
        n_win = min(n_win, max_windows)
    tot_nll, tot_n = 0.0, 0
    hit_next, hit_curr = 0, 0
    # position buckets (prereg §1): token positions 1-32 / 33-128 / 129-256
    # within eval windows, identical for every arm (state reset per window)
    bounds = (min(32, S), min(128, S), S)
    b_nll = [0.0, 0.0, 0.0]
    b_n = [0, 0, 0]
    WB = 128        # window batch: 16 left the GPU 8x underutilized (review 5)
    for lo in range(0, n_win * S, S * WB):
        hi = min(lo + S * WB, n_win * S)
        xs = np.stack([eval_ids[s:s + S] for s in range(lo, hi, S)])
        ys = np.stack([eval_ids[s + 1:s + S + 1] for s in range(lo, hi, S)])
        x = torch.from_numpy(xs.astype(np.int64)).to(device)
        y = torch.from_numpy(ys.astype(np.int64)).to(device)
        if is_recurrent(model):
            logits, _ = model(x, model.init_state(x.shape[0], device), counter)
        else:
            logits, _ = model(x, None, counter)
        nll_pos = F.cross_entropy(logits.reshape(-1, config.V), y.reshape(-1),
                                  reduction="none").view(y.shape)
        tot_nll += float(nll_pos.sum())
        tot_n += y.numel()
        p0 = 0
        for i, p1 in enumerate(bounds):
            b_nll[i] += float(nll_pos[:, p0:p1].sum())
            b_n[i] += nll_pos[:, p0:p1].numel()
            p0 = p1
        am = logits.argmax(-1)
        hit_next += int((am == y).sum())
        hit_curr += int((am == x).sum())
    model.train()
    return {"bpc": tot_nll / tot_n / math.log(2), "n_chars": tot_n,
            "p_argmax_next": hit_next / tot_n, "p_argmax_curr": hit_curr / tot_n,
            "bucket_bpc": [n and s / n / math.log(2) for s, n in zip(b_nll, b_n)],
            "bucket_n": b_n}


def spike_density(model, x, device):
    """Spikes per token on one batch, for death detection. Cheap, no grad."""
    if not hasattr(model, "arm") or model.arm not in ("T2", "R2"):
        return None
    c = RuntimeCounter(model)
    with torch.no_grad():
        model(x.to(device), model.init_state(x.shape[0], device), c)
    r = c.report()
    return r["spikes_l1"]["value"] + r["spikes_l2"]["value"]


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    import os
    os.replace(path + ".tmp", path)


def train_run(model, steps, next_batch, device, label, eval_fn=None,
              eval_every=250, log=None, ckpt_path=None, resume=None,
              grid=(), grid_dir=None, skip_batches=0):
    """The one training loop. next_batch() -> (x, y) on cpu. eval_fn(model) ->
    dict merged into history. Returns result dict (never raises on death —
    a death is a recorded outcome, not a crash).

    Protocol additions (prereg §3, ops review M1): atomic {model, opt, step}
    checkpoint to ckpt_path every CKPT_EVERY steps; resume from a loaded ckpt
    dict; model-state saves at registered grid steps into grid_dir (the LOCF
    basis for §7 and the C3 interpolation grid). Callers doing resume must
    fast-forward the batcher themselves (exact: skip step*1 batches) and pass
    skip_batches for the record."""
    import os
    model.to(device).train()
    spiking = hasattr(model, "arm") and model.arm in ("T2", "R2")
    peak = config.LR_SPIKE if spiking else config.LR_DENSE
    peak = config.ARM_OVERRIDES.get(getattr(model, "arm", ""), {}).get(
        "lr", peak)
    opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=config.WD)
    start_step = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        start_step = resume["step"]
    hist = list(resume.get("history", [])) if resume else []
    death = None
    silent_streak = 0
    t0 = time.time()
    tokens_per_step = None
    for step in range(start_step, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak)
        x, y = next_batch()
        x, y = x.to(device), y.to(device)
        tokens_per_step = x.numel()
        opt.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, x, y)
        if not math.isfinite(loss):
            # verdict BEFORE opt.step(): the saved checkpoint stays pre-NaN
            # and usable for post-mortem (review m8)
            death = {"kind": "nan", "step": step + 1}
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
        opt.step()
        if spiking and step % 10 == 0 and step > 0:
            d = spike_density(model, x[: min(8, x.shape[0])], device)
            silent_streak = silent_streak + 10 if d < config.SILENCE_SPIKES_PER_TOKEN else 0
            if silent_streak >= config.SILENCE_WINDOW_STEPS:
                death = {"kind": "silent", "step": step + 1,
                         "spikes_per_token": d}
                break
        at_grid = (step + 1) in grid
        if step % eval_every == eval_every - 1 or step == steps - 1 or at_grid:
            row = {"step": step + 1, "train_loss_bits": loss / math.log(2),
                   "lr": lr_at(step, steps, peak),
                   "elapsed_s": round(time.time() - t0, 1),
                   "tokens_per_s": round((step + 1 - start_step) * tokens_per_step
                                         / (time.time() - t0), 1)}
            if eval_fn is not None:
                row.update(eval_fn(model))
            stop = bool(row.pop("_stop", False))
            if at_grid:
                row["grid"] = True
                if grid_dir:
                    save_atomic(model.state_dict(),
                                os.path.join(grid_dir, f"grid_{label}_{step + 1}.pt"))
            hist.append(row)
            if log:
                log({"label": label, **row})
            if stop:
                break
        if ckpt_path and (step + 1) % config.CKPT_EVERY == 0:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": step + 1, "history": hist}, ckpt_path)
    return {"label": label, "history": hist, "death": death,
            "steps_done": step + 1,
            "resumed_from": start_step or None,
            "tokens_per_step": qty(tokens_per_step or 0, "step"),
            "wall_s": round(time.time() - t0, 1)}


def accounting_pass(model, eval_ids, device):
    """Dedicated instrumented pass, measured TWICE by genuinely independent
    paths: RuntimeCounter (add_step calls inside the forward loop) vs
    HookRecorder (module pre-hooks + shape-derived tokens). The first draft's
    static check was algebraically circular — it consumed the runtime
    counter's own rates and could never fire (both reviews, finding 1)."""
    c = RuntimeCounter(model)
    h = HookRecorder(model)
    try:
        r = eval_bpc(model, eval_ids, device, max_windows=64, counter=c)
    finally:
        h.detach()
    rep = c.report()
    hooked = h.synops_per_token()
    ok = agreement(rep["synops"]["value"], hooked)
    return {"runtime": rep, "hooked_synops": qty(hooked, "token"),
            "hook_tokens": qty(h.tokens, "run"),
            "counters_agree_5pct": bool(ok),
            "efficiency_status": "OK" if ok else "EMBARGOED",
            "acct_eval_bpc": r["bpc"]}
