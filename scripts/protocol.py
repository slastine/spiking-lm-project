"""Protocol pod entry (prereg v1.0). Env: ARMS, SEEDS, STEPS (default 6000),
LABEL, DEADLINE_S, OUT. Idempotent and resumable:
- a tag with a complete results file is never re-run (container restarts safe)
- mid-run ckpt every CKPT_EVERY steps -> ckpt dir (network volume if mounted
  at /workspace, else /probe_out); resume picks it up and fast-forwards the
  batcher exactly
- state_dict saved IMMEDIATELY post-train, BEFORE final eval/accounting
  (ops review M2: the epilogue must never be able to destroy trained weights)
- registered grid evals with state saves (prereg §3)
- test span [10.0M,10.5M) touched exactly once, at the final checkpoint,
  results sealed into the row (prereg §1, §4)
"""
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, "/pkg/src" if os.path.isdir("/pkg/src") else
                os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from stage2 import config, data, frozen
from stage2.dense import build
from stage2.probes import (generate, make_induction_probes, novelty_8gram,
                           score_induction)
from stage2.train import (accounting_pass, eval_bpc, save_atomic,
                          seed_everything, train_run)

OUT = os.environ.get("OUT", "/probe_out")
LABEL = os.environ.get("LABEL", "protocol")
STEPS = int(os.environ.get("STEPS", config.PROTOCOL_STEPS))
ARMS = os.environ.get("ARMS", "").split()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "").split(",") if s]
DEADLINE_S = int(os.environ.get("DEADLINE_S", "0"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
CKPT_DIR = ("/workspace/stage2_ckpts" if os.path.isdir("/workspace")
            else OUT)
os.makedirs(CKPT_DIR, exist_ok=True)


def write(name, obj):
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, os.path.join(OUT, name))


def progress(stage, rows, extra=None):
    write("progress.json", {"stage": stage, "label": LABEL, "gate": "protocol",
                            "device": DEV, "rows": rows, **(extra or {})})


def main():
    assert ARMS and SEEDS, "protocol requires explicit ARMS and SEEDS"
    rows = []
    progress("booting", rows)
    ids = data.load_corpus(os.path.join(OUT, "tinystories_12mib.bin"))
    train_ids = data.span(ids, "train")
    val_ids = data.span(ids, "val")
    unigram = data.unigram_bpc(train_ids, val_ids)
    probes_x, probes_spans, probes_sha = make_induction_probes(val_ids)
    progress("running", rows, {"unigram_bpc": unigram,
                               "induction_probe_sha": probes_sha})
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"p_{arm}_s{seed}"
            res_path = os.path.join(OUT, f"results_{tag}.json")
            if os.path.exists(res_path):
                try:
                    prev = json.load(open(res_path))
                    if prev.get("complete"):
                        rows.append(prev)
                        continue            # idempotency: never re-run
                except Exception:
                    pass
            row = {"arm": arm, "seed": seed, "unigram_bpc": unigram,
                   "steps": STEPS, "complete": False}
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
                ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{tag}.pt")
                resume = None
                if os.path.exists(ckpt_path):
                    try:
                        resume = torch.load(ckpt_path, map_location=DEV)
                        batcher.skip(resume["step"])
                        row["resumed_at"] = resume["step"]
                    except Exception:
                        resume = None       # corrupt ckpt: start clean

                def eval_fn(m):
                    return {f"val_{k}": v for k, v in
                            eval_bpc(m, val_ids, DEV, max_windows=128).items()}

                res = train_run(model, STEPS, batcher.next, DEV, tag,
                                eval_fn=eval_fn, eval_every=config.EVAL_EVERY,
                                log=lambda r: progress("running", rows + [r]),
                                ckpt_path=ckpt_path, resume=resume,
                                grid=config.GRID, grid_dir=OUT)
                # M2: weights to disk BEFORE any epilogue can fail
                save_atomic(model.state_dict(),
                            os.path.join(OUT, f"{tag}_state.pt"))
                row.update({k: res[k] for k in
                            ("history", "death", "steps_done", "wall_s",
                             "resumed_from")})
                if not res["death"]:
                    row["final_val"] = eval_bpc(model, val_ids, DEV)
                    # single registered test-span touch (prereg §1/§4)
                    row["final_test"] = eval_bpc(model, data.span(ids, "test"),
                                                 DEV)
                    if arm in ("T2", "R2"):
                        row["accounting"] = accounting_pass(model,
                                                            val_ids[:16640], DEV)
                    row["induction"] = score_induction(model, probes_x,
                                                       probes_spans, DEV)
                    row["induction_sha"] = probes_sha
                    gens, novs = [], []
                    test_ids = data.span(ids, "test")
                    for gi in range(8):
                        p0 = 1000 * gi + seed           # seed-pinned prefixes
                        prompt = list(test_ids[p0:p0 + 32])
                        g = generate(model, prompt, 512, DEV, gen_seed=gi)
                        novs.append(novelty_8gram(g, train_ids))
                        if gi < 3:
                            gens.append("".join(frozen.VOCAB[c] for c in g))
                    row["generations_first3"] = gens
                    row["novelty_8gram"] = novs
                row["complete"] = True
            except Exception:
                row["error"] = traceback.format_exc()[-2000:]
            rows.append(row)
            progress("running", rows, {"unigram_bpc": unigram})
            write(f"results_{tag}.json", row)
    progress("complete", rows, {"unigram_bpc": unigram,
                                "induction_probe_sha": probes_sha})


if __name__ == "__main__":
    main()
