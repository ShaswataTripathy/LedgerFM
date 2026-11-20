"""Builds notebooks/LedgerFM_quickstart.ipynb (run: python notebooks/build_notebook.py; execute with nbclient)."""
import nbformat as nbf, pathlib
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = [
md("""# LedgerFM quickstart: transaction embeddings for fraud detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShaswataTripathy/LedgerFM/blob/main/notebooks/LedgerFM_quickstart.ipynb)

[LedgerFM](https://huggingface.co/tripathyShaswata/LedgerFM) is an open foundation model for transaction sequences. It reads each card's or account's history and returns one 256-dimensional embedding per transaction.

This notebook runs the whole recipe on 1.85M public, synthetic card transactions (Sparkov, CC0):
1. embed every transaction with the pretrained model
2. train an XGBoost baseline on simple per-transaction fields
3. add the LedgerFM embeddings as features
4. blend a LedgerFM-only score with XGBoost, choosing the weight on validation

All splits are temporal: train on the past, choose on validation, report on the future (test).

**Runtime:** use a GPU (Runtime → Change runtime type → T4). On CPU, set `SAMPLE_CARDS` below to keep it quick."""),
code("!pip -q install ledgerfm xgboost scikit-learn"),
code("""import time, numpy as np, polars as pl, torch, xgboost as xgb
from huggingface_hub import hf_hub_download
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from ledgerfm import load, embed, Blend

SAMPLE_CARDS = None   # e.g. 200 on CPU; None = all ~1,000 cards
print("GPU:", torch.cuda.is_available())"""),
md("## 1. Data\nOne row per transaction, in LedgerFM's input schema (`txn_id, ts, account, counterparty, amount, currency, category, ...`). `split` is a 60/20/20 temporal split."),
code("""df = pl.read_parquet(hf_hub_download("tripathyShaswata/ledgerfm-demo-sparkov", "sparkov.parquet", repo_type="dataset"))
if SAMPLE_CARDS:
    keep = df["account"].unique().sort().head(SAMPLE_CARDS)
    df = df.filter(pl.col("account").is_in(keep.implode()))
print(df.shape)
df.group_by("split").agg(pl.len().alias("rows"), pl.col("label").sum().alias("frauds"), pl.col("ts").min().alias("from"), pl.col("ts").max().alias("to")).sort("from")"""),
md("## 2. Embed every transaction\nEach embedding sees only the card's **earlier** transactions, so there is no look-ahead."),
code("""fm = load()   # downloads the v0.1 weights (106 MB) from Hugging Face
t = time.time()
E = embed(fm, df)
print(E.shape, f"{time.time() - t:.0f}s")"""),
md("## 3. Features\nThe baseline uses only per-transaction fields (amount, time, category, state). LedgerFM adds history-aware embeddings, reduced to 32 dimensions with PCA fitted on training rows."),
code("""base = df.select(
    "txn_id", "split", "label",
    pl.col("amount").abs().log1p().alias("log_amount"),
    pl.col("ts").dt.hour().alias("hour"), pl.col("ts").dt.weekday().alias("weekday"),
    pl.col("category").cast(pl.Categorical).to_physical().alias("category"),
    pl.col("location").cast(pl.Categorical).to_physical().alias("state"),
).join(E, on="txn_id")

emb_cols = [c for c in E.columns if c.startswith("emb")]
X_emb = base.select(emb_cols).to_numpy().astype(np.float32)
tr = (base["split"] == "train").to_numpy(); va = (base["split"] == "val").to_numpy(); te = (base["split"] == "test").to_numpy()
mu = X_emb[tr].mean(0)
_, _, Vt = np.linalg.svd(X_emb[tr][np.random.default_rng(0).choice(tr.sum(), min(200_000, tr.sum()), replace=False)] - mu, full_matrices=False)
Z = (X_emb - mu) @ Vt[:32].T
X_base = base.select("log_amount", "hour", "weekday", "category", "state").to_numpy().astype(np.float32)
y = base["label"].to_numpy()
print(X_base.shape, Z.shape)"""),
md("## 4. Models (validation picks everything; test is scored at the end)"),
code("""def fit_xgb(X):
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist")
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return m.predict_proba(X)[:, 1]

p_xgb = fit_xgb(X_base)                          # baseline
p_xgb_emb = fit_xgb(np.hstack([X_base, Z]))       # + LedgerFM embeddings
head = LogisticRegression(max_iter=3000, C=1.0).fit(X_emb[tr], y[tr])
p_fm = head.predict_proba(X_emb)[:, 1]            # LedgerFM-only score (linear head on frozen embeddings)
blend = Blend().fit(y[va], p_fm[va], p_xgb[va])   # weight chosen on validation
p_blend = blend.predict(p_fm, p_xgb)
print("blend weight on LedgerFM:", blend.w)"""),
code("""rows = []
for name, p in [("XGBoost, per-transaction fields", p_xgb), ("LedgerFM embeddings + linear head", p_fm),
                ("XGBoost + LedgerFM embeddings", p_xgb_emb), ("Blend: LedgerFM head x XGBoost", p_blend)]:
    rows.append({"model": name, "val PR-AUC": round(average_precision_score(y[va], p[va]), 3),
                 "test PR-AUC": round(average_precision_score(y[te], p[te]), 3)})
pl.DataFrame(rows)"""),
md("""## What this shows, and what it doesn't

- The embeddings carry the card's **history**: what it usually buys, where, how often. Per-transaction fields can't see any of that.
- This notebook uses frozen embeddings for speed. Fine-tuning the whole model (`models/finetune.py` in the [repo](https://github.com/ShaswataTripathy/LedgerFM)) is stronger: on this dataset it reaches test PR-AUC 0.991, against 0.946 for XGBoost with 29 engineered history features.
- **Limits:** LedgerFM was pretrained mostly on synthetic data, including Sparkov's training period, so this dataset is not an unseen test. A model pretrained without Sparkov still scored 0.978. On short account histories and with few labels, XGBoost alone is often as good or better. The blend picks its weight on validation, so it falls back to XGBoost when the embeddings don't help.

**Your own data:** produce the same columns (`txn_id, ts, account, counterparty, amount, currency`, plus optional `category, channel, location, ...`) and call `embed(fm, df)`. Or from the command line: `ledgerfm embed transactions.parquet out.parquet --pca 32`."""),
]
nb = nbf.v4.new_notebook(cells=C, metadata={"accelerator": "GPU", "colab": {"provenance": []},
     "kernelspec": {"name": "python3", "display_name": "Python 3"}})
p = pathlib.Path(__file__).with_name("LedgerFM_quickstart.ipynb")
nbf.write(nb, p); print("wrote", p)
