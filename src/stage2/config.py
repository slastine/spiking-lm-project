"""Stage 2 constants. Anything that would silently produce a different experiment if
changed is a constant here, not a default argument scattered through call sites."""

V = 64                  # folded vocabulary (frozen.py)
D_EMB = 64              # embedding dim == input channels for spiking arms
W = 12                  # simulation timesteps per token window (spiking arms)
SEQ = 256               # training sequence length, chars
BATCH = 32              # protocol batch
TBPTT = 64              # truncation chunk, tokens (recurrent arms)
PROTOCOL_STEPS = 6000   # defines "100%"; gate 1 runs 10%
GATE1_STEPS = 600
GATE0_STEPS = 5000
GATE0_BATCH = 50
GATE0_SEQS = 100        # memorization corpus size
GATE0_SEQ_LEN = 65      # 64 predicted targets
GATE0_PASS_BPC = 0.5
EVAL_EVERY = 250        # gate 0 threshold checks
GATE1_EVAL_EVERY = 100

# spiking cell
N_HID = 768
N_LAYERS = 2
V_TH = 1.0
ALPHA = 0.95            # membrane decay per step (tau ~ 20 steps)
SURROGATE_SLOPE = 10.0
LP_ALPHA = 0.8          # low-pass filter for readout spike counts

# dense arms
GPT_LAYERS, GPT_HEADS, GPT_D = 6, 5, 160
LSTM_LAYERS, LSTM_H = 2, 384
# GPT-FM: compute-matched transformer (rho = 21.95M/2.345M = 9.36;
# 8L x 7H x d448 -> 21.13M MACs/token = 0.96*rho, within the +/-10% band)
GPTFM_LAYERS, GPTFM_HEADS, GPTFM_D = 8, 7, 448
LSTM12_MICRO = 12       # unroll-depth control: micro-steps per token

# protocol
CKPT_EVERY = 500        # mid-run crash checkpoints (operational)
GRID = (600, 1200, 2400, 3600, 4800, 6000)  # evaluation checkpoints

# optimization (frozen for gates). LR_DENSE/LR_SPIKE,
# GPT_DROPOUT and SURROGATE_SLOPE are the tuning-grid axes;
# tune.py overrides them per grid point, protocol runs take the winners.
LR_DENSE = 1e-3
LR_SPIKE = 2e-3
GPT_DROPOUT = 0.0

# Tuning-grid winners (pinned 2026-08-09, before any protocol run, from
# runs/tuneG and runs/tuneT2; seed 99, full grids published). ONLY the arms
# named here take winners — controls inherit the gate recipes above, and
# GPTFM inherits GPTU's winner, per the analysis plan.
ARM_OVERRIDES = {
    "GPTU": {"lr": 2e-3, "dropout": 0.0},
    "GPTFM": {"lr": 2e-3, "dropout": 0.0},
    "T2": {"lr": 4e-3, "slope": 5.0},
}
WD = 0.01
CLIP = 1.0
WARMUP = 200

# cheap arms first: if the pod-side timeout lands mid-schedule it must truncate
# the expensive spiking runs, never delete the dense comparators (review 4)
ARMS = ("GPTU", "GPTQ", "LSTMU", "LSTM12", "GPTFM", "R2", "T2")

# death criteria
SILENCE_SPIKES_PER_TOKEN = 1.0
SILENCE_WINDOW_STEPS = 50
