"""Tuning-parity grids, frozen here. Env: ARMS in {GPTU, T2}
(one per invocation or space-sep), LABEL, OUT. Seed 99, 1,200 steps each
point, scored on capped val; full grid published; winner is advisory — the
protocol launch pins it in config.ARM_OVERRIDES.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/pkg/src" if os.path.isdir("/pkg/src") else
                os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from stage2 import config, data
from stage2.dense import build
from stage2.train import eval_bpc, seed_everything, train_run

OUT = os.environ.get("OUT", "/probe_out")
LABEL = os.environ.get("LABEL", "tune")
ARMS = os.environ.get("ARMS", "GPTU T2").split()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 1200
SEED = 99                    # excluded from all analysis

GRIDS = {
    "GPTU": [{"LR_DENSE": lr, "GPT_DROPOUT": dr}
             for lr in (5e-4, 1e-3, 2e-3) for dr in (0.0, 0.1)],
    "T2": [{"LR_SPIKE": lr, "SURROGATE_SLOPE": sl}
           for lr in (1e-3, 2e-3, 4e-3) for sl in (5.0, 10.0)],
}
DEFAULTS = {k: getattr(config, k) for k in
            ("LR_DENSE", "GPT_DROPOUT", "LR_SPIKE", "SURROGATE_SLOPE")}


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, os.path.join(OUT, name))


def main():
    config.ARM_OVERRIDES = {}     # grids vary these knobs; winners must not pin
    rows = []
    write("progress.json", {"stage": "booting", "label": LABEL, "rows": rows})
    ids = data.load_corpus(os.path.join(OUT, "tinystories_12mib.bin"))
    train_ids, val_ids = data.span(ids, "train"), data.span(ids, "val")
    for arm in ARMS:
        for point in GRIDS[arm]:
            row = {"arm": arm, "point": point}
            try:
                for k, v in point.items():
                    setattr(config, k, v)
                seed_everything(1000 + SEED)
                model = build(arm)
                batcher = data.OrderedBatcher(train_ids, config.SEQ,
                                              config.BATCH, SEED)
                res = train_run(model, STEPS, batcher.next, DEV,
                                f"tune_{arm}", eval_every=400,
                                log=lambda r: write("progress.json",
                                                    {"stage": "running",
                                                     "label": LABEL,
                                                     "rows": rows + [r]}))
                row["death"] = res["death"]
                row["wall_s"] = res["wall_s"]
                if not res["death"]:
                    ev = eval_bpc(model, val_ids, DEV, max_windows=512)
                    row["val_bpc"] = ev["bpc"]
            except Exception:
                row["error"] = traceback.format_exc()[-1500:]
            finally:
                for k in point:                      # restore frozen defaults
                    setattr(config, k, DEFAULTS[k])
            rows.append(row)
            write("progress.json", {"stage": "running", "label": LABEL,
                                    "rows": rows})
            write(f"results_tune_{arm}.json",
                  {"arm": arm, "grid": [r for r in rows if r["arm"] == arm]})
    # winner per arm, plus median-over-grid (selection-correction reporting)
    summary = {}
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm and "val_bpc" in r]
        if g:
            best = min(g, key=lambda r: r["val_bpc"])
            vals = sorted(r["val_bpc"] for r in g)
            summary[arm] = {"winner": best["point"],
                            "winner_bpc": best["val_bpc"],
                            "median_bpc": vals[len(vals) // 2],
                            "n_scored": len(g), "n_dead": len(
                                [r for r in rows if r["arm"] == arm
                                 and r.get("death")])}
    write("results_tune_summary.json", summary)
    write("progress.json", {"stage": "complete", "label": LABEL,
                            "rows": rows, "summary": summary})


if __name__ == "__main__":
    main()
