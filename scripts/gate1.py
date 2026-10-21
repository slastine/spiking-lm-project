"""Gate 1 pod entry: screening, 3 seeds/arm x 10% protocol steps on TinyStories.

Env: ARMS, SEEDS (comma-sep, default 0,1,2), STEPS (default 600), LABEL, OUT.
Per run: results_g1_<arm>_s<seed>.json + state dict. progress.json carries the
running decision inputs (val BPC vs unigram, echo, density, death).
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
from stage2.train import accounting_pass, eval_bpc, seed_everything, train_run

OUT = os.environ.get("OUT", "/probe_out")
LABEL = os.environ.get("LABEL", "gate1")
STEPS = int(os.environ.get("STEPS", config.GATE1_STEPS))
ARMS = os.environ.get("ARMS", " ".join(config.ARMS)).split()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
DEADLINE_S = int(os.environ.get("DEADLINE_S", "0"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, os.path.join(OUT, name))


def progress(stage, rows, extra=None):
    write("progress.json", {"stage": stage, "label": LABEL, "gate": 1,
                            "device": DEV, "rows": rows, **(extra or {})})


def main():
    rows = []
    progress("booting", rows)
    ids = data.load_corpus(os.path.join(OUT, "tinystories_12mib.bin"))
    train_ids, val_ids = data.span(ids, "train"), data.span(ids, "val")
    unigram = data.unigram_bpc(train_ids, val_ids)
    progress("running", rows, {"unigram_bpc": unigram})
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"g1_{arm}_s{seed}"
            row = {"arm": arm, "seed": seed, "unigram_bpc": unigram}
            if DEADLINE_S and time.time() - T0 > DEADLINE_S:
                row["skipped"] = "out_of_time"
                rows.append(row)
                write(f"results_{tag}.json", row)
                continue
            try:
                seed_everything(1000 + seed)
                model = build(arm)
                row["params"] = sum(p.numel() for p in model.parameters())
                batcher = data.OrderedBatcher(train_ids, config.SEQ,
                                              config.BATCH, seed)

                def eval_fn(m):
                    # capped mid-training eval; full-val eval happens at the end
                    return {f"val_{k}": v for k, v in
                            eval_bpc(m, val_ids, DEV, max_windows=128).items()}

                res = train_run(model, STEPS, batcher.next, DEV, tag,
                                eval_fn=eval_fn,
                                eval_every=config.GATE1_EVAL_EVERY,
                                log=lambda r: progress("running", rows + [r]))
                row.update({k: res[k] for k in
                            ("history", "death", "steps_done", "wall_s")})
                if not res["death"]:
                    row["final"] = eval_bpc(model, val_ids, DEV)  # full val span
                    if arm in ("T2", "R2"):
                        row["accounting"] = accounting_pass(model,
                                                            val_ids[:16640], DEV)
                torch.save(model.state_dict(),
                           os.path.join(OUT, f"{tag}_state.pt"))
                if DEV == "cuda":
                    row["peak_mem_gb"] = round(
                        torch.cuda.max_memory_allocated() / 2**30, 2)
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                row["error"] = traceback.format_exc()[-2000:]
            rows.append(row)
            progress("running", rows, {"unigram_bpc": unigram})
            write(f"results_{tag}.json", row)
    progress("complete", rows, {"unigram_bpc": unigram})


if __name__ == "__main__":
    main()
