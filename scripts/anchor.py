"""Reference-anchor run (prereg §2.2): our GPT harness at the published
nanoGPT shakespeare_char configuration (6L x 6H x d384, dropout 0.2, ctx 256,
~10.6M params, batch 64, 5000 iters) on tiny-shakespeare. Must land within
0.10 BPC of the published 1.4697 nats = 2.121 BPC or the protocol is blocked
(the transformer arm's external positive control).
"""
import hashlib
import json
import math
import os
import sys
import traceback
import urllib.request

sys.path.insert(0, "/pkg/src" if os.path.isdir("/pkg/src") else
                os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from stage2 import config
from stage2.dense import build
from stage2.train import batch_loss, seed_everything, train_run

OUT = os.environ.get("OUT", "/probe_out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
TARGET_BPC = 1.4697 / math.log(2)          # published nanoGPT val loss
TOL = 0.10


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, os.path.join(OUT, name))


def main():
    row = {"target_bpc": TARGET_BPC, "tol": TOL, "pass": False}
    try:
        raw = urllib.request.urlopen(URL, timeout=120).read()
        row["data_sha"] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        chars = sorted(set(text))
        row["vocab"] = len(chars)              # 65 for tiny-shakespeare
        ctoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([ctoi[c] for c in text], dtype=np.int64)
        n = int(len(ids) * 0.9)                # nanoGPT's 90/10 split
        tr, va = ids[:n], ids[n:]
        # nanoGPT shakespeare_char config, on our harness:
        config.ARM_OVERRIDES = {}          # anchor pins its own recipe below
        config.V = len(chars)
        config.GPT_LAYERS, config.GPT_HEADS, config.GPT_D = 6, 6, 384
        config.GPT_DROPOUT = 0.2
        config.LR_DENSE = 1e-3
        config.WARMUP = 100
        seed_everything(1337)
        model = build("GPTU")
        row["params"] = sum(p.numel() for p in model.parameters())
        rng = np.random.default_rng(1337)

        def next_batch():
            ix = rng.integers(0, len(tr) - config.SEQ - 1, size=64)
            x = np.stack([tr[i:i + config.SEQ] for i in ix])
            y = np.stack([tr[i + 1:i + config.SEQ + 1] for i in ix])
            return torch.from_numpy(x), torch.from_numpy(y)

        def val_bpc(m):
            m.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for lo in range(0, len(va) - config.SEQ - 1, config.SEQ * 64):
                    ix = range(lo, min(lo + config.SEQ * 64,
                                       len(va) - config.SEQ - 1), config.SEQ)
                    if not ix:
                        break
                    x = torch.from_numpy(np.stack([va[i:i + config.SEQ]
                                                   for i in ix])).to(DEV)
                    y = torch.from_numpy(np.stack([va[i + 1:i + config.SEQ + 1]
                                                   for i in ix])).to(DEV)
                    loss, _ = batch_loss(m, x, y)
                    tot += loss * y.numel()
                    cnt += y.numel()
            m.train()
            return tot / cnt / math.log(2)

        res = train_run(model, 5000, next_batch, DEV, "anchor",
                        eval_fn=lambda m: {"val_bpc": val_bpc(m)},
                        eval_every=500,
                        log=lambda r: write("progress.json",
                                            {"stage": "running", "row": r}))
        row["history"] = res["history"]
        best = min(h["val_bpc"] for h in res["history"] if "val_bpc" in h)
        row["best_val_bpc"] = best
        row["pass"] = abs(best - TARGET_BPC) <= TOL or best < TARGET_BPC
        row["death"] = res["death"]
    except Exception:
        row["error"] = traceback.format_exc()[-2000:]
    write("results_anchor.json", row)
    write("progress.json", {"stage": "complete", "row": row})


if __name__ == "__main__":
    main()
