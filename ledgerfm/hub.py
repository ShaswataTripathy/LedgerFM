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


REQUIRED = ["txn_id", "ts", "account", "counterparty", "amount"]
_STR = ["account", "counterparty", "currency", "currency_recv", "category", "channel", "location", "instrument",
        "src_bank", "dst_bank", "errors"]


def canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise user input to the canonical schema: ids/categoricals as strings (CSV readers often turn card numbers
    into integers), missing optional columns added as null (currency defaults to USD), ts parsed if given as text."""
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"missing required columns {miss}; required: {REQUIRED}")
    ts = pl.col("ts").str.to_datetime(strict=False) if df["ts"].dtype == pl.String else pl.col("ts")
    cols = [pl.col("txn_id").cast(pl.Int64), ts.cast(pl.Datetime("us")).alias("ts"), pl.col("amount").cast(pl.Float64),
            (pl.col("amount_recv").cast(pl.Float64) if "amount_recv" in df.columns else pl.lit(None, pl.Float64)).alias("amount_recv")]
    for c in _STR:
        cols.append((pl.col(c).cast(pl.String) if c in df.columns else pl.lit("USD" if c == "currency" else None, pl.String)).alias(c))
    out = df.with_columns(cols)
    if out["ts"].null_count():
        raise ValueError("could not parse some ts values; use e.g. 2024-01-31 13:45:00")
    return out


def embed(fm, df: pl.DataFrame, bs=64) -> pl.DataFrame:
    """df: canonical rows (txn_id, ts, account, counterparty, amount, currency, ... see README).
    Returns txn_id + emb0..emb{d-1} (float32), one row per input transaction."""
    ctx = context_columns(canonical(df))
    ints, fl = tokenize(ctx)
    st = _Store(ints, fl, ctx["account"].to_numpy())
    H = np.full((st.N, fm.c.d), np.nan, np.float32)
    dev = next(fm.parameters()).device.type

    def fn(h, idx):
        H[idx] = h.cpu().numpy()
    run_rows(fm, st, fm.c.max_len, {0}, fn, bs=bs, device=dev)
    return pl.DataFrame({"txn_id": ctx["txn_id"]}).with_columns(
        [pl.Series(f"emb{j}", H[:, j]) for j in range(fm.c.d)])
