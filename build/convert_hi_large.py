"""Fresh, never-used test world: first 11 days (2022-08-01 .. 2022-08-11) of IBM AML HI-Large.

Only the first ~11.5 GB of HI-Large_Trans.csv is on disk (download stopped for disk space); the file is
time-ordered at day granularity, and 2022-08-12 is reached at ~20% of the file, so the first 30M lines
hold every row before the cut-off. Same canonical schema and 60/20/20 temporal split as the other sets.
Output: data/processed/aml_hi_large.parquet
"""
import sys, time, pathlib, datetime as dt
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import convert

NAME, CUT, LINES = "aml_hi_large", dt.datetime(2022, 8, 12), 30_000_000

if __name__ == "__main__":
    t0 = time.time()
    lf, _, _ = convert.aml(NAME)
    df = (lf.head(LINES).filter(pl.col("ts") < CUT)
          .select([pl.col(c).cast(t) for c, t in convert.CANON.items()]).collect())
    late = lf.head(LINES).select((pl.col("ts") >= CUT).sum()).collect().item()
    df = convert.add_splits(df)
    print(df.group_by("split").agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"),
                                   pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1")).sort("t0"), flush=True)
    df = df.sort("account", "ts", "txn_id")
    df.write_parquet(convert.PROC / f"{NAME}.parquet", compression="zstd", row_group_size=1_000_000, statistics=True)
    print(f"rows={df.height:,} accounts={df['account'].n_unique():,} pos={int(df['label'].sum()):,} "
          f"(rows >= cut-off in first {LINES:,} lines: {late:,})  {time.time()-t0:.0f}s", flush=True)
