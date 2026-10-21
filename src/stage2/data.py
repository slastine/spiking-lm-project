"""Data: download-verify-fold against frozen.py, batching, gate-0 corpus.

Every artifact of this module is checked against a frozen hash before use. The pod
path and the local path run the identical code; there is no "pod variant" to drift.
"""
import hashlib
import urllib.request

import numpy as np
import torch

from . import config, frozen

_DIGITS = "0123456789"


def fold(text: str) -> bytes:
    """Fold raw text (with <|endoftext|> already replaced by \\x00) to vocab ids."""
    ctoi = {c: i for i, c in enumerate(frozen.VOCAB)}
    out = bytearray()
    for ch in text:
        ch = frozen.ALIAS.get(ch, ch)
        if ch in _DIGITS:
            ch = "#"
        out.append(ctoi.get(ch, frozen.UNK_ID))
    return bytes(out)


def load_corpus(cache_path: str = "/probe_out/tinystories_12mib.bin") -> np.ndarray:
    """Download (or reuse cached) pinned byte range, verify BOTH hashes, return ids.

    Raises on any mismatch: silently different data must be impossible.
    """
    import os
    if os.path.exists(cache_path):
        raw = open(cache_path, "rb").read()
    else:
        req = urllib.request.Request(
            frozen.HF_URL,
            headers={"Range": f"bytes=0-{frozen.RAW_BYTES - 1}",
                     "User-Agent": "stage2/0.1"})
        rsp = urllib.request.urlopen(req, timeout=120)
        if rsp.status != 206:
            # a server ignoring Range would stream the full ~1.9GB file
            raise RuntimeError(f"expected 206 partial content, got {rsp.status}")
        raw = rsp.read()
        with open(cache_path, "wb") as f:
            f.write(raw)
    if len(raw) != frozen.RAW_BYTES:
        raise RuntimeError(f"raw byte count {len(raw)} != frozen {frozen.RAW_BYTES}")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != frozen.RAW_SHA256:
        raise RuntimeError(f"raw sha {sha} != frozen {frozen.RAW_SHA256}")
    ids = fold(raw.decode("utf-8", errors="ignore").replace("<|endoftext|>", "\x00"))
    if len(ids) != frozen.FOLDED_CHARS:
        raise RuntimeError(f"folded length {len(ids)} != frozen {frozen.FOLDED_CHARS}")
    fsha = hashlib.sha256(ids).hexdigest()
    if fsha != frozen.FOLDED_SHA256:
        raise RuntimeError(f"folded sha {fsha} != frozen {frozen.FOLDED_SHA256}")
    return np.frombuffer(ids, dtype=np.uint8)


def span(ids: np.ndarray, which: str) -> np.ndarray:
    lo, hi = {"train": frozen.TRAIN_SPAN, "val": frozen.VAL_SPAN,
              "test": frozen.TEST_SPAN}[which]
    return ids[lo:hi]


class OrderedBatcher:
    """Deterministic per-seed data order, shared across arms by seed index.

    Start offsets are drawn without replacement from a permutation of all valid
    positions in the span, reshuffled (deterministically, from the same seed
    sequence) when exhausted. Two arms constructed with the same seed see the
    same (x, y) streams in the same order.
    """

    def __init__(self, span_ids: np.ndarray, seq: int, batch: int, seed: int):
        self.ids = span_ids
        self.seq, self.batch = seq, batch
        self.rng = np.random.default_rng(np.random.SeedSequence([20260808, seed]))
        self.n_pos = len(span_ids) - seq - 1
        if self.n_pos <= 0:
            raise ValueError("span shorter than sequence length")
        self._perm = self.rng.permutation(self.n_pos)
        self._i = 0

    def skip(self, n_batches: int):
        """Exact fast-forward for resume (ops review M1): protocol draws
        (192,000) never exhaust the 9.5M-position permutation, so advancing
        the index reproduces the stream a fresh batcher would have produced."""
        if (n_batches + 1) * self.batch > len(self._perm):
            raise ValueError("skip would cross a reshuffle boundary")
        self._i = n_batches * self.batch

    def next(self):
        if self._i + self.batch > len(self._perm):
            self._perm = self.rng.permutation(self.n_pos)
            self._i = 0
        starts = self._perm[self._i:self._i + self.batch]
        self._i += self.batch
        x = np.stack([self.ids[s:s + self.seq] for s in starts])
        y = np.stack([self.ids[s + 1:s + self.seq + 1] for s in starts])
        return (torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(y.astype(np.int64)))


def gate0_corpus(train_ids: np.ndarray):
    """Fixed 100x65 memorization corpus, deterministic."""
    rng = np.random.default_rng(np.random.SeedSequence([20260808, 999]))
    starts = rng.choice(len(train_ids) - config.GATE0_SEQ_LEN,
                        size=config.GATE0_SEQS, replace=False)
    seqs = np.stack([train_ids[s:s + config.GATE0_SEQ_LEN] for s in starts])
    x = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    y = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    return x, y


def unigram_bpc(train_ids: np.ndarray, eval_ids: np.ndarray) -> float:
    """Measured context-free baseline: add-1-smoothed unigram fit on train,
    scored on eval. This is the C1 reference; it is computed, never assumed."""
    counts = np.bincount(train_ids, minlength=config.V).astype(np.float64) + 1.0
    logp = np.log2(counts / counts.sum())
    return float(-logp[eval_ids].mean())
