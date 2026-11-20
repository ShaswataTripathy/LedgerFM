"""Incoming view: the same transactions, grouped by RECEIVER instead of sender.

Roles are swapped (account <-> counterparty, src_bank <-> dst_bank, sent <-> received amount/currency),
then the normal tokenizer runs. Row t of <ds>__in = transaction t seen from the receiver's side, with the
receiver's earlier incoming transactions as history. Same txn_id / split / label arrays as the sender view.
Output: data/tokens/<ds>__in   Usage: python -u build/tokenize_inview.py ds [ds ...]
"""
import sys, pathlib
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "build"))
import tokenize_all

SWAP = [("account", "counterparty"), ("src_bank", "dst_bank")]


def swap_roles(df: pl.DataFrame) -> pl.DataFrame:
    ren = {}
    for a, b in SWAP:
        ren[a], ren[b] = b, a
    df = df.rename(ren)
    has = pl.col("amount_recv").is_not_null()
    return df.with_columns(
        pl.when(has).then(pl.col("amount_recv")).otherwise(pl.col("amount")).alias("amount"),
        pl.when(has).then(pl.col("amount")).otherwise(pl.col("amount_recv")).alias("amount_recv"),
        pl.when(pl.col("currency_recv").is_not_null()).then(pl.col("currency_recv")).otherwise(pl.col("currency")).alias("currency"),
        pl.when(pl.col("currency_recv").is_not_null()).then(pl.col("currency")).otherwise(pl.col("currency_recv")).alias("currency_recv"),
    )


if __name__ == "__main__":
    orig = tokenize_all.pl.read_parquet
    for name in sys.argv[1:]:
        tokenize_all.pl.read_parquet = lambda p, _o=orig: swap_roles(_o(p))
        # tokenize_all.run reads PROC/<name>.parquet and writes TOK/<name>; point the output at <name>__in
        src = tokenize_all.PROC / f"{name}.parquet"
        link = tokenize_all.PROC / f"{name}__in.parquet"
        if not link.exists():
            import os
            os.link(src, link)  # hard link: no extra disk
        try:
            tokenize_all.run(f"{name}__in")
        finally:
            link.unlink()
        tokenize_all.pl.read_parquet = orig
