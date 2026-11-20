"""Convert raw CSVs to the LedgerFM canonical schema (Parquet).

Outputs
  data/processed/<ds>.parquet  canonical rows, sorted by (account, ts, txn_id)
  data/interim/<ds>.raw.parquet all original columns, for strong tabular baselines
Splits (written as columns, decided before any modelling):
  split      temporal 60/20/20 by transaction time (IBM's recommended AML split)
  acct_fold  hash(account) % 10, for the account-disjoint second view
The CSVs are deleted only after the Parquet row count matches the CSV.
Usage: python -u build/convert.py [--keep-csv] [ds ...]
"""
import sys, time, pathlib
import polars as pl
pl.Config.set_ascii_tables(True)
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data/raw", ROOT / "data/interim", ROOT / "data/processed"

CANON = {
    "ds": pl.String, "txn_id": pl.Int64, "ts": pl.Datetime("us"),
    "account": pl.String, "counterparty": pl.String,
    "amount": pl.Float64, "currency": pl.String,
    "amount_recv": pl.Float64, "currency_recv": pl.String,
    "category": pl.String, "channel": pl.String, "location": pl.String,
    "instrument": pl.String, "src_bank": pl.String, "dst_bank": pl.String,
    "errors": pl.String, "label": pl.Int8,
}
NULL_S, NULL_F = pl.lit(None, pl.String), pl.lit(None, pl.Float64)


def aml(name):
    f = next((RAW / name).glob("*_Trans.csv"))
    cols = ["Timestamp", "From Bank", "Account", "To Bank", "Account2", "Amount Received",
            "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    lf = pl.scan_csv(f, has_header=True, new_columns=cols, infer_schema=False)
    out = lf.with_row_index("txn_id").select(
        pl.lit(name).alias("ds"), pl.col("txn_id").cast(pl.Int64),
        pl.col("Timestamp").str.strptime(pl.Datetime("us"), "%Y/%m/%d %H:%M").alias("ts"),
        (pl.col("From Bank") + "_" + pl.col("Account")).alias("account"),
        (pl.col("To Bank") + "_" + pl.col("Account2")).alias("counterparty"),
        pl.col("Amount Paid").cast(pl.Float64).alias("amount"),
        pl.col("Payment Currency").alias("currency"),
        pl.col("Amount Received").cast(pl.Float64).alias("amount_recv"),
        pl.col("Receiving Currency").alias("currency_recv"),
        NULL_S.alias("category"), pl.col("Payment Format").alias("channel"), NULL_S.alias("location"),
        NULL_S.alias("instrument"), pl.col("From Bank").alias("src_bank"), pl.col("To Bank").alias("dst_bank"),
        NULL_S.alias("errors"), pl.col("Is Laundering").cast(pl.Int8).alias("label"),
    )
    return out, [f], None  # canonical already holds every raw column


def ibm_card(name):
    f = RAW / name / "credit_card_transactions-ibm_v2.csv"
    lf = pl.scan_csv(f, infer_schema=False).with_row_index("txn_id")
    ts = pl.concat_str([pl.col("Year"), pl.col("Month").str.zfill(2), pl.col("Day").str.zfill(2), pl.col("Time")], separator=" ")
    out = lf.select(
        pl.lit(name).alias("ds"), pl.col("txn_id").cast(pl.Int64),
        ts.str.strptime(pl.Datetime("us"), "%Y %m %d %H:%M").alias("ts"),
        ("u" + pl.col("User")).alias("account"), pl.col("Merchant Name").alias("counterparty"),
        pl.col("Amount").str.replace(r"\$", "").cast(pl.Float64).alias("amount"),
        pl.lit("USD").alias("currency"), NULL_F.alias("amount_recv"), NULL_S.alias("currency_recv"),
        pl.col("MCC").alias("category"), pl.col("Use Chip").alias("channel"),
        pl.coalesce(pl.col("Merchant State"), pl.col("Merchant City")).alias("location"),
        ("u" + pl.col("User") + "_c" + pl.col("Card")).alias("instrument"),
        NULL_S.alias("src_bank"), NULL_S.alias("dst_bank"), pl.col("Errors?").alias("errors"),
        (pl.col("Is Fraud?") == "Yes").cast(pl.Int8).alias("label"),
    )
    side = [RAW / name / x for x in ("sd254_cards.csv", "sd254_users.csv", "User0_credit_card_transactions.csv")]
    return out, [f] + side, lf


def sparkov(name):
    fs = [RAW / name / "fraudTrain.csv", RAW / name / "fraudTest.csv"]
    lf = pl.concat([pl.scan_csv(f, infer_schema=False).drop("") for f in fs]).with_row_index("txn_id")
    out = lf.select(
        pl.lit(name).alias("ds"), pl.col("txn_id").cast(pl.Int64),
        pl.col("trans_date_trans_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S").alias("ts"),
        pl.col("cc_num").alias("account"), pl.col("merchant").alias("counterparty"),
        pl.col("amt").cast(pl.Float64).alias("amount"), pl.lit("USD").alias("currency"),
        NULL_F.alias("amount_recv"), NULL_S.alias("currency_recv"),
        pl.col("category"), NULL_S.alias("channel"), pl.col("state").alias("location"),
        pl.col("cc_num").alias("instrument"), NULL_S.alias("src_bank"), NULL_S.alias("dst_bank"),
        NULL_S.alias("errors"), pl.col("is_fraud").cast(pl.Int8).alias("label"),
    )
    # raw copy for baselines: drop direct personal fields of the synthetic people
    raw = lf.drop("first", "last", "street", "trans_num")
    return out, fs, raw


def paysim(name):
    f = next((RAW / name).glob("PS_*.csv"))
    lf = pl.scan_csv(f, infer_schema=False).with_row_index("txn_id")
    base = pl.datetime(2017, 1, 1)  # PaySim 'step' = 1 simulated hour; the start date is arbitrary
    out = lf.select(
        pl.lit(name).alias("ds"), pl.col("txn_id").cast(pl.Int64),
        (base + pl.duration(hours=pl.col("step").cast(pl.Int64))).cast(pl.Datetime("us")).alias("ts"),
        pl.col("nameOrig").alias("account"), pl.col("nameDest").alias("counterparty"),
        pl.col("amount").cast(pl.Float64), pl.lit("UNK").alias("currency"),
        NULL_F.alias("amount_recv"), NULL_S.alias("currency_recv"),
        pl.col("type").alias("category"), pl.col("type").alias("channel"), NULL_S.alias("location"),
        NULL_S.alias("instrument"), NULL_S.alias("src_bank"), NULL_S.alias("dst_bank"),
        NULL_S.alias("errors"), pl.col("isFraud").cast(pl.Int8).alias("label"),
    )
    return out, [f], lf


def elliptic(name):
    d = RAW / name / "elliptic_bitcoin_dataset"
    feats = pl.read_csv(d / "elliptic_txs_features.csv", has_header=False)
    feats.columns = ["txId", "timestep"] + [f"f{i}" for i in range(1, feats.width - 1)]
    INT.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(INT / "elliptic_features.parquet")
    pl.read_csv(d / "elliptic_txs_classes.csv").write_parquet(INT / "elliptic_classes.parquet")
    pl.read_csv(d / "elliptic_txs_edgelist.csv").write_parquet(INT / "elliptic_edges.parquet")
    return None, list(d.glob("*.csv")), None


def ieee_cis(name):
    """IEEE-CIS (Vesta) - EVALUATION ONLY (Kaggle competition rules; never in released weights).
    No account id is given: use the standard reconstructed uid = card1 + addr1 + first-use day (day - D1).
    TransactionDT is seconds from an undisclosed reference; 2017-12-01 is the usual assumption."""
    ft, fi = RAW / name / "train_transaction.csv", RAW / name / "train_identity.csv"
    tr = pl.scan_csv(ft, infer_schema=False); idt = pl.scan_csv(fi, infer_schema=False)
    lf = tr.join(idt, on="TransactionID", how="left")
    dt = pl.col("TransactionDT").cast(pl.Int64)
    day = dt // 86400
    d1 = pl.col("D1").cast(pl.Float64).round(0).cast(pl.Int64)
    na = lambda c: pl.col(c).fill_null("na")
    uid = pl.concat_str([na("card1"), na("addr1"), (day - d1).cast(pl.String).fill_null("na")], separator="_")
    out = lf.select(
        pl.lit(name).alias("ds"), pl.col("TransactionID").cast(pl.Int64).alias("txn_id"),
        (pl.datetime(2017, 12, 1) + pl.duration(seconds=dt)).cast(pl.Datetime("us")).alias("ts"),
        uid.alias("account"), NULL_S.alias("counterparty"),
        pl.col("TransactionAmt").cast(pl.Float64).alias("amount"), pl.lit("USD").alias("currency"),
        NULL_F.alias("amount_recv"), NULL_S.alias("currency_recv"),
        pl.col("ProductCD").alias("category"), pl.col("card4").alias("channel"), pl.col("addr1").alias("location"),
        pl.concat_str([na("card1"), na("card6")], separator="_").alias("instrument"),
        NULL_S.alias("src_bank"), NULL_S.alias("dst_bank"), NULL_S.alias("errors"),
        pl.col("isFraud").cast(pl.Int8).alias("label"),
    )
    raw = lf.rename({"TransactionID": "txn_id"}).drop("isFraud", "TransactionDT")
    return out, [ft, fi], raw


BUILDERS = {"aml_hi_small": aml, "aml_li_small": aml, "aml_hi_medium": aml, "aml_li_medium": aml,
            "ibm_card": ibm_card, "sparkov": sparkov, "paysim": paysim, "elliptic": elliptic,
            "ieee_cis": ieee_cis}


def add_splits(df):
    # temporal 60/20/20 by row count in time order (ties broken by txn_id)
    df = df.sort("ts", "txn_id")
    n = df.height
    pos = pl.int_range(0, n, eager=False)
    df = df.with_columns(
        pl.when(pos < int(0.6 * n)).then(pl.lit("train"))
          .when(pos < int(0.8 * n)).then(pl.lit("val")).otherwise(pl.lit("test")).alias("split"),
        (pl.col("account").hash(seed=17) % 10).cast(pl.Int8).alias("acct_fold"),
    )
    return df


def csv_rows(paths):
    return sum(pl.scan_csv(p, infer_schema=False).select(pl.len()).collect().item() for p in paths)


def run(name, keep_csv):
    t0 = time.time()
    done = PROC / f"{name}.parquet"
    if done.exists() or (name == "elliptic" and (INT / "elliptic_features.parquet").exists()):
        print(f"[skip] {name}", flush=True)
        return
    print(f"[conv] {name}", flush=True)
    canon, csvs, raw = BUILDERS[name](name)
    INT.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)
    n_out = None
    if canon is not None:
        df = canon.select([pl.col(c).cast(t) for c, t in CANON.items()]).collect()
        df = add_splits(df)
        s = df.group_by("split").agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"),
                                     pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1")).sort("t0")
        print(s, flush=True)
        df = df.sort("account", "ts", "txn_id")
        df.write_parquet(done, compression="zstd", row_group_size=1_000_000, statistics=True)
        n_out = df.height
        print(f"  rows={n_out:,} accounts={df['account'].n_unique():,} pos={int(df['label'].sum()):,} "
              f"null_ts={df['ts'].null_count()} null_amt={df['amount'].null_count()}", flush=True)
        del df
    if raw is not None:
        raw.sink_parquet(INT / f"{name}.raw.parquet", compression="zstd")
    if name == "ibm_card":
        for p in csvs[1:3]:
            pl.read_csv(p, infer_schema=False).write_parquet(INT / f"ibm_card_{p.stem}.parquet")
    # verify, then delete the CSVs
    main_csvs = csvs if name not in ("ibm_card", "ieee_cis") else csvs[:1]
    if n_out is not None:
        n_csv = csv_rows(main_csvs)
        if n_csv != n_out:
            print(f"  [WARN] row mismatch csv={n_csv:,} parquet={n_out:,}; keeping CSVs", flush=True)
            return
    if not keep_csv:
        for p in csvs:
            p.unlink()
    print(f"[done] {name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    keep = "--keep-csv" in args
    names = [a for a in args if not a.startswith("--")] or list(BUILDERS)
    for n in names:
        run(n, keep)
