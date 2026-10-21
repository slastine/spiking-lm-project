"""Gate 0 pod entry: per-arm memorization positive control + throughput.

Env: ARMS (space-sep), STEPS (default 5000), LABEL, OUT (default /probe_out).
Writes results_g0_<arm>.json + g0_<arm>_state.pt per arm, progress.json summary.
Every failure mode is a recorded verdict, not a crash: a broken arm must show up
as FAIL in the artifact, never as an absent file.
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

from stage2 import config, data
from stage2.dense import build
from stage2.train import accounting_pass, batch_loss, seed_everything, train_run

OUT = os.environ.get("OUT", "/probe_out")
LABEL = os.environ.get("LABEL", "gate0")
STEPS = int(os.environ.get("STEPS", config.GATE0_STEPS))
ARMS = os.environ.get("ARMS", " ".join(config.ARMS)).split()
DEADLINE_S = int(os.environ.get("DEADLINE_S", "0"))    # 0 = none
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()


def write(name, obj):
    # atomic: a reap reading mid-write must never see a torn file (review 9)
    tmp = os.path.join(OUT, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, os.path.join(OUT, name))


def progress(stage, rows):
    write("progress.json", {"stage": stage, "label": LABEL, "gate": 0,
                            "device": DEV, "rows": rows})


def corpus_bpc(model, x, y):
    with torch.no_grad():
        model.eval()
        loss, _ = batch_loss(model, x, y)
        model.train()
    return loss / math.log(2)


def probe_step_seconds(model, n=3):
    """Time full train steps at the gate1 shape (BATCH x SEQ) on real data."""
    import torch.nn.functional as F  # noqa: F401  (kept local; probe only)
    xp = torch.randint(0, config.V, (config.BATCH, config.SEQ), device=DEV)
    yp = torch.randint(0, config.V, (config.BATCH, config.SEQ), device=DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # no weight change
    batch_loss(model, xp, yp)                            # warm kernels
    opt.zero_grad(set_to_none=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        batch_loss(model, xp, yp)
    if DEV == "cuda":
        torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)
    return {"s_per_step": round((time.time() - t) / n, 4),
            "shape": [config.BATCH, config.SEQ]}


def main():
    rows = []
    progress("booting", rows)
    ids = data.load_corpus(os.path.join(OUT, "tinystories_12mib.bin"))
    train_ids = data.span(ids, "train")
    x_all, y_all = data.gate0_corpus(train_ids)
    x_all, y_all = x_all.to(DEV), y_all.to(DEV)
    x_cpu, y_cpu = x_all.cpu(), y_all.cpu()
    for arm in ARMS:
        row = {"arm": arm, "pass": False, "pass_step": None}
        if DEADLINE_S and time.time() - T0 > DEADLINE_S:
            # a run that never started must still appear in the artifact
            row["skipped"] = "out_of_time"
            rows.append(row)
            write(f"results_g0_{arm}.json", row)
            continue
        try:
            seed_everything(0)
            model = build(arm)
            row["params"] = sum(p.numel() for p in model.parameters())
            i = {"i": 0}

            def next_batch():
                lo = (i["i"] * config.GATE0_BATCH) % config.GATE0_SEQS
                i["i"] += 1
                return (x_cpu[lo:lo + config.GATE0_BATCH],
                        y_cpu[lo:lo + config.GATE0_BATCH])

            def eval_fn(m):
                bpc = corpus_bpc(m, x_all, y_all)
                # early stop on pass: the criterion is "<0.5 within 5,000
                # steps" — training past the crossing is pure spend with no
                # evidentiary value (live-run amendment, 2026-08-08)
                return {"corpus_bpc": bpc, "_stop": bpc < config.GATE0_PASS_BPC}

            res = train_run(model, STEPS, next_batch, DEV, f"g0_{arm}",
                            eval_fn=eval_fn, eval_every=config.EVAL_EVERY,
                            log=lambda r: progress("running", rows + [r]))
            final_bpc = corpus_bpc(model, x_all, y_all)
            row["final_bpc"] = final_bpc
            row["death"] = res["death"]
            row["history"] = res["history"]
            row["wall_s"] = res["wall_s"]
            row["tokens_per_s"] = (res["history"][-1]["tokens_per_s"]
                                   if res["history"] else None)
            for h in res["history"]:
                if h.get("corpus_bpc", 99) < config.GATE0_PASS_BPC:
                    row["pass_step"] = h["step"]
                    break
            # 'held' vs 'touched' are different claims (review m9)
            row["pass_final"] = bool(final_bpc < config.GATE0_PASS_BPC)
            row["pass_ever"] = row["pass_step"] is not None or row["pass_final"]
            row["pass"] = row["pass_final"] and not res["death"]
            if DEV == "cuda":
                row["peak_mem_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 2)
                torch.cuda.reset_peak_memory_stats()
            if arm in ("T2", "R2"):
                row["accounting"] = accounting_pass(
                    model, data.span(ids, "val")[:16640], DEV)
            # cost projection from a probe at the ACTUAL gate1 shape — gate0's
            # 50x64 throughput does not transfer to 32x256 for launch-bound
            # spiking arms (review 6)
            row["probe_gate1_shape"] = probe_step_seconds(model)
            spb = row["probe_gate1_shape"]["s_per_step"]
            row["proj_gate1_h_per_seed"] = round(config.GATE1_STEPS * spb / 3600, 3)
            row["proj_protocol_h_per_seed"] = round(
                config.PROTOCOL_STEPS * spb / 3600, 2)
            torch.save(model.state_dict(), os.path.join(OUT, f"g0_{arm}_state.pt"))
        except Exception:
            row["error"] = traceback.format_exc()[-2000:]
        rows.append(row)
        progress("running", rows)
        write(f"results_g0_{arm}.json", row)
    progress("complete", rows)


if __name__ == "__main__":
    main()
