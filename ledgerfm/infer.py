"""Inference over every row of a TokenStore with full causal history (overlapping windows)."""
import numpy as np
import torch
from .data import pack, make_batch


def scored_windows(store, L, splits):
    """Eval windows whose scored part contains rows from `splits`. Returns starts, lens, score_from."""
    st, ln, sf = store.eval_chunks(L)
    sp = np.isin(np.asarray(store.split), list(splits))
    cum = np.r_[0, np.cumsum(sp)]
    keep = (cum[st + ln] - cum[st + sf]) > 0
    return st[keep], ln[keep], sf[keep]


@torch.no_grad()
def run_rows(model, store, L, splits, fn, bs=128, device="cuda"):
    """Call fn(h [n,d] tensor, row_idx [n] numpy) for every row in `splits`, each scored exactly once."""
    st, ln, sf = scored_windows(store, L, splits)
    rows = pack(ln, L, np.arange(len(st)))
    sp = np.asarray(store.split); want = np.isin(sp, list(splits))
    model.eval()
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        ints, fl, seg, pos, idx = make_batch(store, st, ln, r, L)
        # mark which positions are scored: pos >= score_from of its chunk
        sfrom = np.zeros(idx.shape, np.int64)
        for b, chunks in enumerate(r):
            c = 0
            for ch in chunks:
                sfrom[b, c:c + ln[ch]] = sf[ch]; c += ln[ch]
        idx_np = idx.numpy()
        m = (idx_np >= 0) & (pos.numpy() >= sfrom)
        m[m] &= want[idx_np[m]]
        if not m.any():
            continue
        with torch.autocast(device, dtype=torch.bfloat16):
            h = model.encode(ints.to(device), fl.to(device), seg.to(device), pos.to(device))
        mt = torch.from_numpy(m).to(device)
        fn(h[mt].float(), idx_np[m])
