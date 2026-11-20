"""Command line: ledgerfm embed transactions.parquet embeddings.parquet [--pca 32] [--model tripathyShaswata/LedgerFM]

Input: Parquet or CSV in the canonical schema (see README). Output: Parquet with txn_id + emb* columns.
--pca k reduces to k dimensions (PCA fitted on the input rows), the form that worked best as XGBoost features.
"""
import argparse, sys
import numpy as np, polars as pl


def _read(path):
    return pl.read_csv(path, try_parse_dates=True) if str(path).lower().endswith(".csv") else pl.read_parquet(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ledgerfm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="embed transactions with LedgerFM")
    e.add_argument("input"); e.add_argument("output")
    e.add_argument("--model", default="tripathyShaswata/LedgerFM")
    e.add_argument("--pca", type=int, default=0); e.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from .hub import load, embed
    df = _read(a.input)
    E = embed(load(a.model, device=a.device), df)
    if a.pca:
        X = E.drop("txn_id").to_numpy().astype(np.float32); mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), min(len(X), 200_000), replace=False)] - mu,
                                 full_matrices=False)
        Z = (X - mu) @ Vt[:a.pca].T
        E = pl.DataFrame({"txn_id": E["txn_id"]}).with_columns([pl.Series(f"fm{j}", Z[:, j]) for j in range(a.pca)])
    E.write_parquet(a.output)
    print(f"wrote {E.height:,} rows x {E.width - 1} dims to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
