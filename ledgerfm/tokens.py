"""Canonical rows -> LedgerFM transaction tokens.

Each transaction becomes
  ints   [N, len(INT_FIELDS)]   categorical codes (stable md5 hashing, fixed transforms)
  floats [N, len(FLOAT_FIELDS)] continuous values with fixed (dataset-free) scaling
The transforms use no dataset statistics, so the same code tokenises any new institution's data.
Needs per-row causal context columns; `context_columns(df)` computes them from canonical rows.
"""
import hashlib
import numpy as np
import polars as pl

# field name -> vocabulary size (code 0 is reserved for missing / masked)
INT_FIELDS = {
    "amt_b": 64, "gap_b": 32, "hour": 25, "wday": 8,
    "category": 4096, "channel": 512, "location": 2048, "currency": 256, "currency_recv": 256,
    "errors": 256, "cp": 32768, "flags": 64, "r_n_b": 32, "r_gap_b": 32,
}
FLOAT_FIELDS = ["lamt", "lgap", "r_lnprev", "r_lcnt24", "r_mlamt", "r_has", "s_dev", "s_lcnt24",
                "hsin", "hcos", "dsin", "dcos"]
# pretraining targets: masked-field and next-transaction heads predict these
MASK_TARGETS = ["amt_b", "category", "channel", "location", "currency", "cp", "hour"]
NEXT_TARGETS = ["amt_b", "category", "channel", "gap_b", "flags"]


# currency names -> ISO 4217 (anything else is upper-cased and kept)
ISO = {"US Dollar": "USD", "Euro": "EUR", "Swiss Franc": "CHF", "Yuan": "CNY", "Shekel": "ILS", "Rupee": "INR",
       "UK Pound": "GBP", "Yen": "JPY", "Ruble": "RUB", "Bitcoin": "BTC", "Canadian Dollar": "CAD",
       "Australian Dollar": "AUD", "Mexican Peso": "MXN", "Saudi Riyal": "SAR", "Brazil Real": "BRL"}
# fixed approximate USD per unit (about 2020-21 levels; consistent with the IBM AML generator).
# Only used to put amounts on one scale; unknown currencies count as 1.
USD_RATE = {"USD": 1.0, "EUR": 1.17, "CHF": 1.09, "CNY": 0.149, "ILS": 0.296, "INR": 0.0136, "GBP": 1.29,
            "JPY": 0.0095, "RUB": 0.0129, "BTC": 11900.0, "CAD": 0.758, "AUD": 0.708, "MXN": 0.0473,
            "SAR": 0.267, "BRL": 0.177}


def normalise_currency(s: pl.Series) -> pl.Series:
    s = s.cast(pl.String)
    return s.replace(ISO).str.to_uppercase()


def stable_hash(s: pl.Series, buckets: int, salt: str) -> np.ndarray:
    """md5-based hash into [1, buckets-1]; 0 = null. Stable across machines and library versions."""
    s = s.cast(pl.String)
    uniq = s.drop_nulls().unique()
    codes = np.fromiter((int.from_bytes(hashlib.md5((salt + v).encode()).digest()[:8], "little") % (buckets - 1) + 1
                         for v in uniq), dtype=np.int64, count=len(uniq))
    m = pl.DataFrame({"v": uniq, "c": codes})
    return (pl.DataFrame({"v": s}).join(m, on="v", how="left", maintain_order="left")["c"]
            .fill_null(0).to_numpy().astype(np.int32))


def context_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Causal per-row context from canonical rows (any order). Returns rows sorted by (account, ts, txn_id)."""
    rate = normalise_currency(df["currency"]).replace_strict(USD_RATE, default=1.0, return_dtype=pl.Float64)
    df = df.with_columns((pl.col("amount") * rate).alias("amount_usd"))
    la = pl.col("amount_usd").abs().log1p()
    one = pl.col("amount_usd").is_not_null().cast(pl.Int32)
    rc = df.sort("counterparty", "ts", "txn_id").select(
        "txn_id",
        pl.int_range(pl.len()).over("counterparty").alias("r_n_prev"),
        pl.col("ts").diff().over("counterparty").dt.total_seconds().alias("r_gap_s"),
        ((la.cum_sum().over("counterparty") - la) / pl.int_range(pl.len()).over("counterparty")).alias("r_mean_lamt_prev"),
        one.rolling_sum_by("ts", window_size="24h").over("counterparty").alias("r_cnt_24h"),
    )
    df = df.sort("account", "ts", "txn_id")
    df = df.with_columns(
        pl.col("ts").diff().over("account").dt.total_seconds().alias("s_gap_s"),
        ((la.cum_sum().over("account") - la) / pl.int_range(pl.len()).over("account")).alias("s_mean_lamt_prev"),
        one.rolling_sum_by("ts", window_size="24h").over("account").alias("s_cnt_24h"),
    )
    npair = df.sort("account", "counterparty", "ts", "txn_id").select(
        "txn_id", (pl.int_range(pl.len()).over("account", "counterparty") == 0).cast(pl.Int8).alias("new_pair"))
    df = df.join(npair, on="txn_id", maintain_order="left").join(rc, on="txn_id", maintain_order="left")
    return df


def _logb(x, scale, n):
    """log bucket: 0 = missing, 1.. = floor(log1p(x)*scale)+1 clipped."""
    v = np.floor(np.log1p(np.clip(np.nan_to_num(x, nan=-1), 0, None)) * scale).astype(np.int64) + 1
    v = np.clip(v, 1, n - 1)
    return np.where(np.isnan(x), 0, v).astype(np.int32)


def tokenize(df: pl.DataFrame):
    """df must come from context_columns(). Returns ints [N,F_int] int32, floats [N,F_float] float32."""
    N = df.height
    f = lambda c: df[c].cast(pl.Float64).to_numpy()
    ccy, ccyr = normalise_currency(df["currency"]), normalise_currency(df["currency_recv"])
    amt = f("amount_usd")
    lamt = np.log1p(np.abs(amt))
    gap = f("s_gap_s"); rgap = f("r_gap_s"); rn = f("r_n_prev")
    hour = df["ts"].dt.hour().to_numpy().astype(np.int32); wday = df["ts"].dt.weekday().to_numpy().astype(np.int32)
    flags = (
        (amt < 0).astype(np.int32)
        | (df["account"] == df["counterparty"]).fill_null(False).to_numpy().astype(np.int32) << 1
        | ((ccyr.is_not_null()) & (ccy != ccyr)).fill_null(False).to_numpy().astype(np.int32) << 2
        | (df["src_bank"] != df["dst_bank"]).fill_null(False).to_numpy().astype(np.int32) << 3
        | ((np.abs(amt) % 1) == 0).astype(np.int32) << 4
        | df["new_pair"].cast(pl.Int32).to_numpy() << 5
    )
    ints = np.zeros((N, len(INT_FIELDS)), np.int32)
    col = {k: i for i, k in enumerate(INT_FIELDS)}
    ints[:, col["amt_b"]] = np.clip(np.floor(lamt * 4).astype(np.int64) + 1, 1, 63)
    ints[:, col["gap_b"]] = _logb(gap, 2.0, 32)
    ints[:, col["hour"]] = hour + 1
    ints[:, col["wday"]] = wday  # polars weekday is 1..7
    for k in ("category", "channel", "location", "errors"):
        ints[:, col[k]] = stable_hash(df[k], INT_FIELDS[k], k)
    ints[:, col["currency"]] = stable_hash(ccy, INT_FIELDS["currency"], "currency")
    ints[:, col["currency_recv"]] = stable_hash(ccyr, INT_FIELDS["currency_recv"], "currency")
    ints[:, col["cp"]] = stable_hash(df["counterparty"], INT_FIELDS["cp"], "cp")
    ints[:, col["flags"]] = flags  # 6 bits, 0..63 all valid
    ints[:, col["r_n_b"]] = _logb(rn, 2.0, 32)
    ints[:, col["r_gap_b"]] = _logb(rgap, 2.0, 32)

    rm = f("r_mean_lamt_prev"); sm = f("s_mean_lamt_prev")
    fl = np.stack([
        lamt / 10, np.log1p(np.nan_to_num(np.clip(gap, 0, None), nan=0)) / 15,
        np.log1p(rn) / 10, np.log1p(f("r_cnt_24h")) / 7,
        np.nan_to_num(rm, nan=0) / 10, (~np.isnan(rm)).astype(float),
        np.nan_to_num(lamt - sm, nan=0) / 5, np.log1p(f("s_cnt_24h")) / 7,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * wday / 7), np.cos(2 * np.pi * wday / 7),
    ], 1).astype(np.float32)
    return ints, np.nan_to_num(fl)
