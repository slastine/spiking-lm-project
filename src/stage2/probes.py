"""Fixed probes: generation policy and the induction probe.

Both are frozen here so their stimulus sets and scoring are committed before
any protocol run launches (the freeze-or-demote rule). The induction probe is
planned as an EXPECTED T2 LOSS and publishes either way.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .train import is_recurrent


@torch.no_grad()
def generate(model, prompt_ids, n_chars, device, temperature=1.0, gen_seed=0):
    """Ancestral sampling, temperature 1.0. One sequence.
    Sampling RNG is a dedicated generator so generation never perturbs any
    other random stream."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(20260808 + gen_seed)
    out = list(prompt_ids)
    if is_recurrent(model):
        st = model.init_state(1, device)
        x = torch.tensor([out], dtype=torch.long, device=device)
        logits, st = model(x, st)
        for _ in range(n_chars):
            p = F.softmax(logits[0, -1] / temperature, dim=-1).cpu()
            c = int(torch.multinomial(p, 1, generator=g))
            out.append(c)
            logits, st = model(torch.tensor([[c]], device=device), st)
    else:
        for _ in range(n_chars):
            ctx = torch.tensor([out[-config.SEQ:]], dtype=torch.long,
                               device=device)
            logits, _ = model(ctx, None)
            p = F.softmax(logits[0, -1] / temperature, dim=-1).cpu()
            out.append(int(torch.multinomial(p, 1, generator=g)))
    model.train()
    return out[len(prompt_ids):]


def novelty_8gram(gen_ids, train_ids) -> dict:
    """Fraction of generated 8-grams absent from the training split."""
    train_set = set()
    tb = train_ids.tobytes()
    for i in range(len(tb) - 7):
        train_set.add(tb[i:i + 8])
    gb = bytes(gen_ids)
    grams = [gb[i:i + 8] for i in range(len(gb) - 7)]
    if not grams:
        return {"novel_frac": None, "n_grams": 0}
    novel = sum(1 for x in grams if x not in train_set)
    return {"novel_frac": novel / len(grams), "n_grams": len(grams)}


def make_induction_probes(val_ids, n_per_dist=300):
    """6-char strings repeated at distances {64, 128, 192} inside 256-char
    windows of real val text (generator seed 20260808). Returns
    (windows [N,256] int64, second-occurrence score positions [N,5], sha256)."""
    rng = np.random.default_rng(np.random.SeedSequence([20260808, 7]))
    xs, spans = [], []
    for d in (64, 128, 192):
        for _ in range(n_per_dist):
            start = int(rng.integers(0, len(val_ids) - config.SEQ - 1))
            w = np.array(val_ids[start:start + config.SEQ], dtype=np.int64)
            s = rng.integers(1, config.V, size=6).astype(np.int64)  # avoid EOS
            p1 = int(rng.integers(0, config.SEQ - d - 7))
            p2 = p1 + d
            w[p1:p1 + 6] = s
            w[p2:p2 + 6] = s
            xs.append(w)
            # score next-char predictions for chars 2..6 of the 2nd copy:
            # position p2+k-1's logits predict w[p2+k], k = 1..5
            spans.append(np.arange(p2, p2 + 5))
    xs = np.stack(xs)
    spans = np.stack(spans)
    sha = hashlib.sha256(xs.tobytes()).hexdigest()
    return xs, spans, sha


@torch.no_grad()
def score_induction(model, xs, spans, device, batch=128):
    """Teacher-forced accuracy at the second occurrence, per distance block."""
    model.eval()
    accs = []
    for lo in range(0, len(xs), batch):
        x = torch.from_numpy(xs[lo:lo + batch]).to(device)
        y = torch.roll(x, -1, dims=1)
        st = model.init_state(x.shape[0], device) if is_recurrent(model) else None
        logits, _ = model(x, st)
        am = logits.argmax(-1)
        for b in range(x.shape[0]):
            pos = spans[lo + b]
            accs.append(float((am[b, pos] == y[b, pos]).float().mean()))
    model.train()
    n = len(accs) // 3
    return {"acc_d64": float(np.mean(accs[:n])),
            "acc_d128": float(np.mean(accs[n:2 * n])),
            "acc_d192": float(np.mean(accs[2 * n:])),
            "n_per_dist": n}
