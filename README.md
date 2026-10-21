# Spiking vs. transformer language models

Code and per-seed results for a seven-arm comparison of spiking and
conventional character-level language models on TinyStories. The thesis arm
is a recurrent spiking network whose input is latency-coded (one spike per
channel, information in its timing). It is compared against a rate-coded
twin, LSTMs, and transformers matched on parameters, compute, and
quantization.

## Arms

| arm | model | protocol seeds |
|---|---|---|
| `T2` | 2-layer recurrent LIF spiking net, **latency-coded** input (one spike per channel, timing carries the value), 12 timesteps per character | 10 (s10–s19) |
| `R2` | same spiking core, **rate-coded** input (spike *count* carries the value, no timing information) | 5 (s10–s14) |
| `GPTU` | nanoGPT-style transformer, 6 layers, d=160 (parameter-matched) | 10 (s10–s19) |
| `GPTFM` | compute-matched transformer, 8 layers, d=448 (21.1M multiply-accumulates per character, ~9x `GPTU`) | 3 (s10–s12) |
| `GPTQ` | `GPTU` with W4A4 quantization-aware training | 5 (s10–s14) |
| `LSTMU` | 2-layer LSTM, hidden 384 | 5 (s10–s14) |
| `LSTM12` | `LSTMU` with each character presented for 12 steps (matches the spiking arms' unroll depth) | 5 (s10–s14) |

43 protocol runs in total, 6,000 steps each.

## Layout

- `src/stage2/`: models and training
  - `spiking.py`: `T2`, `R2`
  - `dense.py`: `GPTU`, `GPTFM`, `GPTQ`, `LSTMU`, `LSTM12`
  - `train.py`: one shared training loop and evaluation (`eval_bpc`) for all arms
  - `accounting.py`: operation counting (multiply-accumulates and synaptic ops), two independent counters that must agree
  - `config.py`: hyperparameters, including the tuned per-arm overrides
  - `data.py`, `frozen.py`: pinned corpus, hash checks, and train/val/test splits (`frozen.py` is generated, do not edit)
  - `probes.py`: text generation and the induction probe
- `scripts/`: run entry points and the scorer (see below)
- `tests/`: test suite
- `data/tinystories_12mib.bin`: the pinned corpus, the first 12 MiB of TinyStories (CDLA-Sharing-1.0)
- `runs/`: results (see below)

## Results (`runs/`)

Each `results_*.json` is one run: validation bits per character (BPC) at
every checkpoint, final validation and test BPC (the test split is scored
once, at the end), operation counts with units, the induction probe, and
generation samples.

| folders | stage | seeds |
|---|---|---|
| `anchor/` | reference check: our GPT harness reproduces nanoGPT's published tiny-shakespeare result | n/a |
| `g0/`, `g0new/`, `g0spike/` | gate 0: each arm can memorize a small corpus (positive control) | n/a |
| `g1a/`, `g1b/` | gate 1: screening, 600 steps | 0–2 |
| `tuneG/`, `tuneT2/` | tuning grids: learning rate x dropout (`GPTU`), learning rate x surrogate slope (`T2`) | 99 |
| `pd1/`–`pd3/`, `pr10/`, `pr13/`, `pt10/`–`pt19/` | protocol runs (dense, `R2`, `T2`) | 10–19 |
| `scoring.json` | all protocol statistics, written by `scripts/score.py` | |
| `sparsity_drift.json` | `T2` spike rates and synaptic ops across checkpoints (seeds 10, 14, 18) | |

Screening (0–2), tuning (99), and protocol (10–19) seeds are disjoint by
design. `pt18/results_p_T2_s19.json` is a placeholder for a run that ran out of
time on that pod. The complete `T2` seed-19 run is in `pt19/`. The scorer skips
incomplete runs.

## Checkpoints

Trained weights for all 43 protocol runs and every intermediate checkpoint
(4.0 GB, 323 files, in seven archives): https://doi.org/10.5281/zenodo.23538593

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Tested with Python 3.14, PyTorch 2.10, and NumPy 1.26 (61 tests, ~90 s on CPU).

## Reproducing the statistics

```bash
python scripts/score.py
```

This reads every `runs/*/results_p_*.json` and rewrites `runs/scoring.json`.
The output is identical to the committed file.

## Running an arm

Each entry script is configured by environment variables and writes its
results to `$OUT`:

```bash
mkdir -p out && cp data/tinystories_12mib.bin out/
ARMS=T2 SEEDS=10 OUT=out python scripts/protocol.py
```

- `protocol.py`: full runs. `ARMS` (space-separated), `SEEDS` (comma-separated), `STEPS` (default 6000)
- `gate0.py`, `gate1.py`, `tune.py`, `anchor.py`: the earlier stages above

If the corpus is not in `$OUT`, it is downloaded from a pinned Hugging Face
revision. Either way it is hash-checked before training, and a mismatch is an
error. Runs use CUDA when available. The spiking arms are slow, because each
character is 12 simulation steps.

## License

MIT (see `LICENSE`).
