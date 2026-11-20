"""Packed-sequence batches from data/tokens/<ds>.

A *chunk* is a contiguous slice of one account's time-ordered history (<= L rows).
Chunks are packed into rows of length L; each chunk gets its own segment id.
"""
import pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK = ROOT / "data/tokens"


class TokenStore:
    def __init__(self, name):
        d = TOK / name
        self.name = name
        self.ints = np.load(d / "ints.npy", mmap_mode="r")
        self.floats = np.load(d / "floats.npy", mmap_mode="r")
        self.label = np.load(d / "label.npy", mmap_mode="r")
        self.split = np.load(d / "split.npy", mmap_mode="r")
        self.txn_id = np.load(d / "txn_id.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.N = len(self.label)

    def train_chunks(self, L, rng=None, lo=0, hi=0):
        """Chunks over each account's rows with lo <= split <= hi (split is time-monotone within an account,
        so these rows are contiguous). Long histories start with a random-length first chunk so borders
        move between epochs."""
        s0, s1 = self.offsets[:-1], self.offsets[1:]
        sp = np.asarray(self.split)
        c_lo = np.r_[0, np.cumsum(sp < lo)]; c_in = np.r_[0, np.cumsum((sp >= lo) & (sp <= hi))]
        first = s0 + (c_lo[s1] - c_lo[s0])
        n_ok = c_in[s1] - c_in[s0]
        starts, lens = [], []
        for a, n in zip(first[n_ok > 0], n_ok[n_ok > 0]):
            b = 0
            if n > L:
                b = int(rng.integers(1, L + 1)) if rng is not None else L
                starts.append(a); lens.append(b)
            while b < n:
                e = min(n, b + L)
                starts.append(a + b); lens.append(e - b); b = e
        return np.array(starts, np.int64), np.array(lens, np.int64)

    def eval_chunks(self, L):
        """Overlapping windows (stride L/2) so every scored row sees >= L/2 rows of history.
        Returns starts, lens, score_from (offset inside chunk from which rows are scored)."""
        H = L // 2
        st, ln, sf = [], [], []
        for a, b in zip(self.offsets[:-1], self.offsets[1:]):
            n = b - a
            s = 0
            while True:
                e = min(n, s + L)
                st.append(a + s); ln.append(e - s); sf.append(0 if s == 0 else H)
                if e == n:
                    break
                s += H
        return np.array(st, np.int64), np.array(ln, np.int64), np.array(sf, np.int64)


def pack(lens, L, order):
    """Greedy sequential packing. Returns list of rows, each a list of chunk indices."""
    rows, cur, used = [], [], 0
    for i in order:
        if used + lens[i] > L and cur:
            rows.append(cur); cur, used = [], 0
        cur.append(i); used += lens[i]
    if cur:
        rows.append(cur)
    return rows


def make_batch(store, starts, lens, rows, L, extra=None):
    """rows: list of lists of chunk ids. Returns tensors (ints, floats, seg, pos, idx) with idx=-1 for padding."""
    B = len(rows)
    idx = np.full((B, L), -1, np.int64); seg = np.zeros((B, L), np.int64); pos = np.zeros((B, L), np.int64)
    for r, chunks in enumerate(rows):
        c = 0
        for s_id, ch in enumerate(chunks, 1):
            n = lens[ch]
            idx[r, c:c + n] = np.arange(starts[ch], starts[ch] + n)
            seg[r, c:c + n] = s_id; pos[r, c:c + n] = np.arange(n); c += n
    flat = idx.reshape(-1); ok = flat >= 0
    ints = np.zeros((B * L, store.ints.shape[1]), np.int64); fl = np.zeros((B * L, store.floats.shape[1]), np.float32)
    sel = np.sort(flat[ok]); inv = np.searchsorted(sel, flat[ok])
    ints[ok] = np.asarray(store.ints[sel], np.int64)[inv]; fl[ok] = np.asarray(store.floats[sel], np.float32)[inv]
    t = lambda a: torch.from_numpy(a)
    return t(ints.reshape(B, L, -1)), t(fl.reshape(B, L, -1)), t(seg), t(pos), t(idx)
