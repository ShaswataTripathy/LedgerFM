"""Load released LedgerFM weights and embed transactions given as a polars DataFrame.

    from ledgerfm.hub import load, embed
    fm = load("tripathyShaswata/LedgerFM")          # or a local folder with model.safetensors + config.json
    E = embed(fm, df)                                # df in the canonical schema (see README) -> polars frame

Each output row is the embedding of one transaction given its own fields and the sender's earlier history.
"""
import json, pathlib
import numpy as np, polars as pl, torch

from .model import LedgerFM, Config
from .tokens import context_columns, tokenize
from .infer import run_rows


def load(repo_or_dir="tripathyShaswata/LedgerFM", device=None):
    p = pathlib.Path(repo_or_dir)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = pathlib.Path(snapshot_download(repo_or_dir, allow_patterns=["model.safetensors", "config.json"]))
    from safetensors.torch import load_file
    fm = LedgerFM(Config(**json.load(open(p / "config.json"))["cfg"]))
    fm.load_state_dict(load_file(p / "model.safetensors"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return fm.to(device).eval()


class _Store:
    """In-memory stand-in for data.TokenStore (rows grouped by account, time-ordered)."""
    def __init__(self, ints, floats, account):
        self.ints, self.floats = ints.astype(np.int64), floats.astype(np.float32)
        self.N = len(ints); self.split = np.zeros(self.N, np.int8)
        starts = np.flatnonzero(np.r_[True, account[1:] != account[:-1]])
        self.offsets = np.r_[starts, self.N].astype(np.int64)

    from .data import TokenStore as _T
    eval_chunks = _T.eval_chunks


def embed(fm, df: pl.DataFrame, bs=64) -> pl.DataFrame:
    """df: canonical rows (txn_id, ts, account, counterparty, amount, currency, ... see README).
    Returns txn_id + emb0..emb{d-1} (float32), one row per input transaction."""
    ctx = context_columns(df)
    ints, fl = tokenize(ctx)
    st = _Store(ints, fl, ctx["account"].to_numpy())
    H = np.full((st.N, fm.c.d), np.nan, np.float32)
    dev = next(fm.parameters()).device.type

    def fn(h, idx):
        H[idx] = h.cpu().numpy()
    run_rows(fm, st, fm.c.max_len, {0}, fn, bs=bs, device=dev)
    return pl.DataFrame({"txn_id": ctx["txn_id"]}).with_columns(
        [pl.Series(f"emb{j}", H[:, j]) for j in range(fm.c.d)])
