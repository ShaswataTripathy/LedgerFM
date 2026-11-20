"""Embed transactions with LedgerFM v0.1 and use the embeddings as extra features.

Run from the repo root:  python examples/quickstart.py
Replace `make_demo()` with your own data in the canonical schema (see README, "Input format").
"""
import sys, pathlib, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ledgerfm.hub import load, embed


def make_demo(n=5000, accounts=200, seed=0):
    """Tiny random ledger in the canonical schema (only for checking that the pipeline runs)."""
    r = np.random.default_rng(seed)
    t0 = dt.datetime(2024, 1, 1)
    return pl.DataFrame({
        "txn_id": np.arange(n, dtype=np.int64),
        "ts": [t0 + dt.timedelta(minutes=int(m)) for m in np.sort(r.integers(0, 60 * 24 * 30, n))],
        "account": [f"A{a}" for a in r.integers(0, accounts, n)],
        "counterparty": [f"M{m}" for m in r.integers(0, 50, n)],
        "amount": np.round(r.lognormal(3, 1.2, n), 2),
        "currency": ["USD"] * n,
        "amount_recv": [None] * n, "currency_recv": [None] * n,
        "category": [f"c{c}" for c in r.integers(0, 12, n)],
        "channel": r.choice(["Chip Transaction", "Online Transaction", "Swipe Transaction"], n).tolist(),
        "location": [None] * n, "instrument": [None] * n,
        "src_bank": [None] * n, "dst_bank": [None] * n, "errors": [None] * n,
    }, schema_overrides={"amount_recv": pl.Float64, "currency_recv": pl.String, "location": pl.String,
                         "instrument": pl.String, "src_bank": pl.String, "dst_bank": pl.String, "errors": pl.String})


if __name__ == "__main__":
    fm = load("tripathyShaswata/LedgerFM")       # downloads model.safetensors + config.json once
    df = make_demo()
    E = embed(fm, df)                            # txn_id + emb0..emb255
    print(E.shape)
    # Typical use: PCA to ~32 dims (fit on training rows), join on txn_id, add to your XGBoost features,
    # or fine-tune end to end with models/finetune.py. See README for the measured gains and limits.
