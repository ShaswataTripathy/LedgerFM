"""Two-sided frozen embeddings + XGBoost (probe for the receiver-history idea).

For transaction t (A -> B at time ts) the extra features are, all causal:
  fm*   sender view:   A's outgoing history up to t            (data/emb/<init>/<ds>.parquet, as embed_xgb)
  ib*   receiver view: B's incoming history up to t            (<ds>__in token store, same txn_id)
  ia*   A's incoming history before t   (latest txn received by A with ts < t)
  ob*   B's outgoing history before t   (latest txn sent by B with ts < t)
Usage: python -u models/twoside_xgb.py --ds aml_hi_small --init checkpoints/fm_all/best.pt [--in-init ...]
"""
import argparse, pathlib, sys
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models"))
from embed_xgb import embed
import baseline_xgb


def context(ds, init, in_init, k):
    out, name = embed(ds, init, 32)
    inn, _ = embed(f"{ds}__in", in_init or init, 32)
    pr = pl.read_parquet(ROOT / "data/processed" / f"{ds}.parquet", columns=["txn_id", "ts", "account", "counterparty"])
    tid = pr["txn_id"].dtype
    out = out.with_columns(pl.col("txn_id").cast(tid)); inn = inn.with_columns(pl.col("txn_id").cast(tid))
    oc, ic = [f"fm{j}" for j in range(k)], [f"fm{j}" for j in range(k)]
    ib = inn.select("txn_id", *[pl.col(c).alias(f"ib{j}") for j, c in enumerate(ic)])
    base = pr.join(ib, on="txn_id", how="left").sort("ts")
    # A's incoming state before t: rows where A was the receiver
    rec = pr.join(inn.select("txn_id", *[pl.col(c).alias(f"ia{j}") for j, c in enumerate(ic)]), on="txn_id") \
            .select(pl.col("counterparty").alias("party"), "ts", *[f"ia{j}" for j in range(k)]).sort("ts")
    base = base.join_asof(rec, on="ts", by_left="account", by_right="party", strategy="backward",
                          allow_exact_matches=False).drop("party", strict=False)
    # B's outgoing state before t: rows where B was the sender
    snd = pr.join(out.select("txn_id", *[pl.col(c).alias(f"ob{j}") for j, c in enumerate(oc)]), on="txn_id") \
            .select(pl.col("account").alias("party"), "ts", *[f"ob{j}" for j in range(k)]).sort("ts")
    base = base.join_asof(snd, on="ts", by_left="counterparty", by_right="party", strategy="backward",
                          allow_exact_matches=False).drop("party", strict=False)
    extra = base.drop("ts", "account", "counterparty").join(out, on="txn_id", how="left")
    return extra, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--init", default="checkpoints/fm_all/best.pt")
    ap.add_argument("--in-init", default=""); ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--label-frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    extra, name = context(a.ds, a.init, a.in_init, a.k)
    tf = pl.read_parquet(baseline_xgb.FEAT / f"{a.ds}.parquet", columns=["txn_id"])["txn_id"]
    extra = extra.with_columns(pl.col("txn_id").cast(tf.dtype))
    print(f"[2side] {a.ds}: coverage ib {extra['ib0'].is_not_null().mean():.2f} ia {extra['ia0'].is_not_null().mean():.2f} "
          f"ob {extra['ob0'].is_not_null().mean():.2f}", flush=True)
    baseline_xgb.run(a.ds, False, a.label_frac, extra=extra, prefix=f"xgb+emb2[{name}]" + (f"[in:{pathlib.Path(a.in_init).parent.name}]" if a.in_init else ""), seed=a.seed)


if __name__ == "__main__":
    main()
