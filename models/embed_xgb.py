"""Frozen LedgerFM embeddings + XGBoost: the cheapest way a bank would use the foundation model.

1. Encode every row (train/val/test) with full causal history -> h [N, d] (float16 memmap on disk).
2. PCA to --dims, fitted on a random sample of *train* rows only.
3. Join as extra columns onto the same features as baseline_xgb and train the same XGBoost (val metrics only).

Usage: python -u models/embed_xgb.py --ds aml_hi_small --init checkpoints/fm_all/best.pt [--label-frac 0.01]
Embeddings are cached in data/emb/<init>/<ds>.parquet and reused across label fractions.
"""
import argparse, pathlib, sys, time
import numpy as np, polars as pl, torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models"))
from ledgerfm.data import TokenStore
from ledgerfm.model import LedgerFM, Config
from ledgerfm.infer import run_rows
import baseline_xgb


def embed(ds, init, dims, device="cuda", pca_n=200_000):
    name = pathlib.Path(init).parent.name
    out = ROOT / "data/emb" / name; out.mkdir(parents=True, exist_ok=True)
    f = out / f"{ds}.parquet"
    if f.exists():
        return pl.read_parquet(f), name
    t0 = time.time()
    if device == "cuda":  # share the 8 GB GPU with a concurrent fine-tune: small batches, capped memory
        torch.cuda.set_per_process_memory_fraction(0.2)
    s = torch.load(ROOT / init, map_location="cpu", weights_only=False)
    fm = LedgerFM(Config(**s["cfg"])); fm.load_state_dict(s["model"]); fm.to(device)
    st = TokenStore(ds)
    raw = out / f"{ds}.full.f16"
    H = np.lib.format.open_memmap(raw, mode="w+", dtype=np.float16, shape=(st.N, fm.c.d))
    seen = np.zeros(st.N, bool)

    def fn(h, idx):
        H[idx] = h.cpu().numpy().astype(np.float16); seen[idx] = True
    run_rows(fm, st, fm.c.max_len, {0, 1, 2}, fn, bs=32, device=device)
    print(f"[emb] {ds}: encoded {seen.sum():,}/{st.N:,} rows ({time.time()-t0:.0f}s)", flush=True)

    sp = np.asarray(st.split)
    tr = np.flatnonzero((sp == 0) & seen)
    samp = np.sort(np.random.default_rng(0).choice(tr, min(pca_n, len(tr)), replace=False))
    X = H[samp].astype(np.float32); mu = X.mean(0)
    _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:dims].T.astype(np.float32)
    ev = float((sv[:dims] ** 2).sum() / (sv ** 2).sum())
    Z = np.empty((st.N, dims), np.float32)
    for i in range(0, st.N, 1_000_000):
        Z[i:i + 1_000_000] = (H[i:i + 1_000_000].astype(np.float32) - mu) @ W
    Z[~seen] = np.nan
    del H; raw.unlink()
    df = pl.DataFrame({"txn_id": np.asarray(st.txn_id)}).with_columns(
        [pl.Series(f"fm{j}", Z[:, j]) for j in range(dims)])
    df.write_parquet(f)
    print(f"[emb] {ds}: PCA {fm.c.d}->{dims} (train var explained {ev:.2f}), saved {f.name} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return df, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--init", default="checkpoints/fm_all/best.pt")
    ap.add_argument("--dims", type=int, default=32); ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--canonical-only", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    extra, name = embed(a.ds, a.init, a.dims, a.device)
    tf = baseline_xgb.pl.read_parquet(baseline_xgb.FEAT / f"{a.ds}.parquet", columns=["txn_id"])["txn_id"]
    if extra["txn_id"].dtype != tf.dtype:
        extra = extra.with_columns(pl.col("txn_id").cast(tf.dtype))
    baseline_xgb.run(a.ds, a.canonical_only, a.label_frac, extra=extra, prefix=f"xgb+emb[{name}]", seed=a.seed)


if __name__ == "__main__":
    main()
