"""Tokenise canonical Parquet into memory-mappable arrays for training.

data/tokens/<ds>/ints.npy    int16 [N, F_int]
                 floats.npy  float16 [N, F_float]
                 label.npy   int8, split.npy int8 (0 train, 1 val, 2 test), txn_id.npy int64
                 offsets.npy int64 [n_accounts+1]  rows are grouped by account, time-ordered
Usage: python -u build/tokenize_all.py [ds ...]
"""
import sys, time, pathlib
import numpy as np, polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ledgerfm.tokens import context_columns, tokenize, INT_FIELDS, FLOAT_FIELDS

PROC, TOK = ROOT / "data/processed", ROOT / "data/tokens"


def run(name):
    out = TOK / name
    if (out / "offsets.npy").exists():
        print(f"[skip] {name}", flush=True); return
    t0 = time.time()
    df = context_columns(pl.read_parquet(PROC / f"{name}.parquet"))
    ints, fl = tokenize(df)
    assert ints.max() < 32768 and ints.min() >= 0
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ints.npy", ints.astype(np.int16)); del ints
    np.save(out / "floats.npy", fl.astype(np.float16)); del fl
    np.save(out / "label.npy", df["label"].to_numpy().astype(np.int8))
    np.save(out / "split.npy", df["split"].replace_strict({"train": 0, "val": 1, "test": 2}).to_numpy().astype(np.int8))
    np.save(out / "txn_id.npy", df["txn_id"].to_numpy())
    acc = df["account"].to_numpy()
    starts = np.flatnonzero(np.r_[True, acc[1:] != acc[:-1]])
    np.save(out / "offsets.npy", np.r_[starts, len(acc)].astype(np.int64))
    lens = np.diff(np.r_[starts, len(acc)])
    print(f"[tok ] {name}: N={len(acc):,} accounts={len(starts):,} len p50={np.median(lens):.0f} "
          f"p90={np.percentile(lens, 90):.0f} max={lens.max():,}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["sparkov", "paysim", "ibm_card", "aml_hi_small", "aml_li_small", "aml_hi_medium", "aml_li_medium"]:
        run(n)
