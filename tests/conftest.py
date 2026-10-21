import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stage2 import config  # noqa: E402


@pytest.fixture
def tiny(monkeypatch):
    """Shrink every arm so CPU tests run in seconds. Models read config at
    construction, so patching before build is sufficient."""
    for k, v in dict(N_HID=32, W=4, D_EMB=16, SEQ=32, BATCH=4, TBPTT=8,
                     GPT_D=32, GPT_HEADS=2, GPT_LAYERS=2, LSTM_H=32,
                     LSTM_LAYERS=2, WARMUP=5).items():
        monkeypatch.setattr(config, k, v)
    return config


@pytest.fixture(scope="session")
def corpus_ids():
    """The real pinned corpus if present, else a deterministic synthetic stand-in
    (tests that REQUIRE the real bytes skip without it)."""
    p = ROOT / "data" / "tinystories_12mib.bin"
    if p.exists():
        from stage2 import data
        return data.load_corpus(str(p))
    rng = np.random.default_rng(0)
    return rng.integers(0, config.V, size=200_000).astype(np.uint8)
