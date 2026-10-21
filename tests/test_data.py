"""Data pinning and batching. The known-bad demonstrations are not decoration:
a verification that has only ever passed is not evidence."""
import hashlib
import pathlib

import numpy as np
import pytest

from stage2 import config, data, frozen

ROOT = pathlib.Path(__file__).parent.parent
BIN = ROOT / "data" / "tinystories_12mib.bin"

needs_real = pytest.mark.skipif(not BIN.exists(), reason="pinned bin absent")


@needs_real
def test_fold_matches_frozen_hashes():
    ids = data.load_corpus(str(BIN))
    assert len(ids) == frozen.FOLDED_CHARS
    assert hashlib.sha256(ids.tobytes()).hexdigest() == frozen.FOLDED_SHA256


@needs_real
def test_corrupted_cache_is_refused(tmp_path):
    """KNOWN-BAD: one flipped byte must be a hard failure, not a warning."""
    raw = bytearray(BIN.read_bytes())
    raw[500] ^= 0xFF
    bad = tmp_path / "corrupt.bin"
    bad.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="sha"):
        data.load_corpus(str(bad))


def test_truncated_cache_is_refused(tmp_path):
    bad = tmp_path / "short.bin"
    bad.write_bytes(b"x" * 100)
    with pytest.raises(RuntimeError, match="byte count"):
        data.load_corpus(str(bad))


def test_fold_properties():
    ids = np.frombuffer(data.fold("Hi, mom!\x00 It’s 42°."), dtype=np.uint8)
    v = frozen.VOCAB
    assert v[ids[0]] == "H" and v[ids[1]] == "i"          # case preserved
    assert frozen.EOS_ID in ids                            # EOS survives
    assert v[ids[-3]] == "#" or "#" in [v[i] for i in ids]  # digits fold to #
    assert frozen.UNK_ID in ids                            # degree sign -> UNK
    assert "'" in [v[i] for i in ids]                      # curly quote aliased
    assert ids.max() < config.V


def test_batcher_deterministic_and_seed_distinct(corpus_ids):
    a1 = data.OrderedBatcher(corpus_ids, 32, 4, seed=0)
    a2 = data.OrderedBatcher(corpus_ids, 32, 4, seed=0)
    b = data.OrderedBatcher(corpus_ids, 32, 4, seed=1)
    x1, y1 = a1.next()
    x2, _ = a2.next()
    xb, _ = b.next()
    assert (x1 == x2).all(), "same seed must give identical stream (cross-arm parity)"
    assert not (x1 == xb).all(), "different seeds must differ"
    assert (x1[:, 1:] == y1[:, :-1]).all(), "targets are inputs shifted by one"


def test_gate0_corpus_fixed(corpus_ids):
    x1, y1 = data.gate0_corpus(corpus_ids)
    x2, _ = data.gate0_corpus(corpus_ids)
    assert (x1 == x2).all()
    assert x1.shape == (config.GATE0_SEQS, config.GATE0_SEQ_LEN - 1)
    assert (x1[:, 1:] == y1[:, :-1]).all()


def test_unigram_baseline_sane(corpus_ids):
    b = data.unigram_bpc(corpus_ids[:100_000], corpus_ids[100_000:120_000])
    assert 0.5 < b <= 6.0, "unigram of a 64-symbol stream must be <= log2(64)"


def test_unigram_catches_wrong_base():
    """KNOWN-BAD: a nats/bits confusion (the classic silent factor of ln2=0.69)
    would push a uniform stream's 'BPC' to ~4.16; the correct value is 6.0."""
    rng = np.random.default_rng(1)
    u = rng.integers(0, 64, size=50_000).astype(np.uint8)
    b = data.unigram_bpc(u, u)
    assert abs(b - 6.0) < 0.01
