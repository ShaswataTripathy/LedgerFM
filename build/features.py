"""Causal tabular features for the strong XGBoost baseline.

Every feature uses only the past (or the same timestamp) of the sender or receiver.
Canonical features work on every dataset; raw extras are added per dataset
where the original columns carry signal (Sparkov demographics/geo, PaySim balances).
Output: data/features/<ds>.parquet
Usage: python -u build/features.py [ds ...]
"""
import sys, time, pathlib
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, INT, FEAT = ROOT / "data/processed", ROOT / "data/interim", ROOT / "data/features"
WINDOWS = ["1h", "24h", "7d"]


def side_feats(df, key, prefix):
    """Velocity features for one side (sender=account or receiver=counterparty)."""
    df = df.sort(key, "ts", "txn_id")
    la = pl.col("amount").abs().log1p()
    ex = [
        pl.col("txn_id"),
        (pl.int_range(pl.len()).over(key)).alias(f"{prefix}_n_prev"),
        (pl.col("ts").diff().over(key).dt.total_seconds()).alias(f"{prefix}_gap_s"),
        ((la.cum_sum().over(key) - la) / pl.int_range(pl.len()).over(key)).alias(f"{prefix}_mean_lamt_prev"),
    ]
    for w in WINDOWS:
        ex.append(pl.col("amount").is_not_null().cast(pl.Int32).rolling_sum_by("ts", window_size=w).over(key).alias(f"{prefix}_cnt_{w}"))
        ex.append(la.rolling_sum_by("ts", window_size=w).over(key).alias(f"{prefix}_sumlamt_{w}"))
    return df.select(ex)


def build(name):
    t0 = time.time()
    df = pl.read_parquet(PROC / f"{name}.parquet")
    la = pl.col("amount").abs().log1p()
    base = df.select(
        "ds", "txn_id", "split", "acct_fold", "label", "ts",
        la.alias("lamt"), (pl.col("amount") < 0).cast(pl.Int8).alias("neg_amt"),
        pl.col("ts").dt.hour().alias("hour"), pl.col("ts").dt.weekday().alias("wday"),
        (pl.col("account") == pl.col("counterparty")).cast(pl.Int8).alias("self_txn"),
        (pl.col("currency") != pl.col("currency_recv")).cast(pl.Int8).alias("fx"),
        (pl.col("src_bank") != pl.col("dst_bank")).cast(pl.Int8).alias("cross_bank"),
        *[pl.col(c).cast(pl.Categorical).alias(c) for c in ("category", "channel", "location", "currency", "errors")],
        (pl.col("amount").abs() % 1 == 0).cast(pl.Int8).alias("round_amt"),
    )
    snd = side_feats(df.select("txn_id", "account", "ts", "amount"), "account", "s")
    rcv = side_feats(df.select("txn_id", "counterparty", "ts", "amount"), "counterparty", "r")
    # first time this sender pays this receiver
    pair = df.sort("account", "counterparty", "ts", "txn_id").select(
        "txn_id", (pl.int_range(pl.len()).over("account", "counterparty") == 0).cast(pl.Int8).alias("new_pair"))
    out = base.join(snd, on="txn_id").join(rcv, on="txn_id").join(pair, on="txn_id")
    out = out.with_columns((pl.col("lamt") - pl.col("s_mean_lamt_prev")).alias("s_lamt_dev"))

    if name == "sparkov":
        raw = pl.read_parquet(INT / "sparkov.raw.parquet").select(
            pl.col("txn_id").cast(pl.Int64),
            pl.col("gender").cast(pl.Categorical),
            pl.col("city_pop").cast(pl.Float64),
            pl.col("dob").str.strptime(pl.Date, "%Y-%m-%d").alias("dob"),
            *[pl.col(c).cast(pl.Float64) for c in ("lat", "long", "merch_lat", "merch_long")],
        )
        out = out.join(raw, on="txn_id").with_columns(
            ((pl.col("ts").dt.date() - pl.col("dob")).dt.total_days() / 365.25).alias("age"),
            (((pl.col("lat") - pl.col("merch_lat")) ** 2 + (pl.col("long") - pl.col("merch_long")) ** 2).sqrt()).alias("geo_dist"),
        ).drop("dob", "merch_lat", "merch_long")
    if name == "paysim":
        raw = pl.read_parquet(INT / "paysim.raw.parquet").select(
            pl.col("txn_id").cast(pl.Int64),
            *[pl.col(c).cast(pl.Float64) for c in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")],
        )
        out = out.join(raw, on="txn_id").with_columns(
            (pl.col("newbalanceOrig") + pl.col("amount") - pl.col("oldbalanceOrg")).alias("err_orig"),
            (pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")).alias("err_dest"),
        ).drop("amount")
    if name == "ibm_card":
        raw = pl.read_parquet(INT / "ibm_card.raw.parquet").select(
            pl.col("txn_id").cast(pl.Int64), pl.col("Zip").cast(pl.Float64).alias("zip"),
            pl.col("Merchant City").cast(pl.Categorical).alias("mcity"))
        out = out.join(raw, on="txn_id")
    if name == "ieee_cis":
        # all 400+ anonymised Vesta columns, prefixed x_ (dropped by --canonical-only)
        raw = pl.read_parquet(INT / "ieee_cis.raw.parquet")
        num, cat = [], []
        for c in raw.columns:
            if c == "txn_id":
                continue
            v = raw[c].cast(pl.Float64, strict=False)
            (num if v.null_count() == raw[c].null_count() else cat).append(c)
        raw = raw.select(pl.col("txn_id").cast(pl.Int64), *[pl.col(c).cast(pl.Float64).alias(f"x_{c}") for c in num],
                         *[pl.col(c).cast(pl.Categorical).alias(f"x_{c}") for c in cat])
        out = out.join(raw, on="txn_id")
    FEAT.mkdir(parents=True, exist_ok=True)
    out = out.drop("ts").sort("txn_id")
    out.write_parquet(FEAT / f"{name}.parquet", compression="zstd")
    print(f"[feat] {name}: {out.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or ["sparkov", "paysim", "ibm_card", "aml_hi_small", "aml_li_small", "aml_hi_medium", "aml_li_medium"]
    for n in names:
        build(n)
