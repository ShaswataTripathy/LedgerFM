"""Strong tabular baseline: XGBoost on causal velocity features.

Train on split=train, early-stop on split=val, report val metrics only.
The test split is scored once, at the end, by models/final_test.py (not here).
Usage: python -u models/baseline_xgb.py [ds ...] [--canonical-only] [--label-frac 0.1]
"""
import sys, json, time, pathlib
import numpy as np, polars as pl, xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ledgerfm.metrics import evaluate, label_keep

FEAT, RES = ROOT / "data/features", ROOT / "results"
META = ["ds", "txn_id", "split", "acct_fold", "label"]
RAW_EXTRA = {"gender", "city_pop", "lat", "long", "age", "geo_dist", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "err_orig", "err_dest", "zip", "mcity"}
MAX_NEG = 4_000_000
PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=8,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              max_cat_to_onehot=1, max_bin=256, nthread=-1, seed=0)


def label_subsample(df, frac, seed=0):
    """Low-label regime: keep a fixed random `frac` of labelled training transactions (same rows for every model)."""
    if frac >= 1:
        return df
    return df.filter(pl.Series(label_keep(df["txn_id"].to_numpy(), frac, seed)))


def to_pandas(frame, feats, cats):
    """Polars -> pandas with one shared category list per column (unseen val/test values stay valid)."""
    import pandas as pd
    X = frame.select(feats).to_pandas()
    for c, levels in cats.items():
        X[c] = pd.Categorical(X[c].astype(object), categories=levels)
    return X


def run(name, canonical_only=False, label_frac=1.0, extra=None, prefix="xgb", seed=0):
    """extra: optional polars frame keyed by txn_id (e.g. LedgerFM embeddings) joined in as more features."""
    t0 = time.time()
    df = pl.read_parquet(FEAT / f"{name}.parquet")
    if extra is not None:
        df = df.join(extra, on="txn_id", how="left")
    feats = [c for c in df.columns if c not in META and not (canonical_only and (c in RAW_EXTRA or c.startswith("x_")))
             and df[c].null_count() < df.height and df[c].n_unique() > 1]  # drop empty/constant columns
    tr = label_subsample(df.filter(pl.col("split") == "train"), label_frac, seed)
    va = df.filter(pl.col("split") == "val")
    neg = tr.filter(pl.col("label") == 0)
    w_neg = 1.0
    if neg.height > MAX_NEG:  # subsample negatives, reweight so probabilities stay calibrated-ish
        w_neg = neg.height / MAX_NEG
        neg = neg.sample(MAX_NEG, seed=seed)
    tr = pl.concat([tr.filter(pl.col("label") == 1), neg])
    w = np.where(tr["label"].to_numpy() == 1, 1.0, w_neg)
    cats = {c: df[c].cast(pl.String).drop_nulls().unique().sort().to_list()
            for c in feats if df[c].dtype == pl.Categorical}
    dtr = xgb.DMatrix(to_pandas(tr, feats, cats), label=tr["label"].to_numpy(), weight=w, enable_categorical=True)
    dva = xgb.DMatrix(to_pandas(va, feats, cats), label=va["label"].to_numpy(), enable_categorical=True)
    bst = xgb.train({**PARAMS, "seed": seed}, dtr, num_boost_round=3000, evals=[(dva, "val")],
                    early_stopping_rounds=150, verbose_eval=250)
    s = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    m = evaluate(va["label"].to_numpy(), s)
    tag = f"{prefix}{'_canon' if canonical_only else ''}{'' if label_frac >= 1 else f'_lf{label_frac}'}{f'_s{seed}' if seed else ''}"
    m.update(ds=name, model=tag, split="val", seed=seed, best_iter=bst.best_iteration, n_feats=len(feats),
             train_pos=int(tr["label"].sum()), secs=round(time.time() - t0))
    imp = bst.get_score(importance_type="gain")
    m["top_feats"] = [k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:8]]
    out = RES / "baselines"; out.mkdir(parents=True, exist_ok=True)
    bst.save_model(out / f"{name}_{tag}.ubj")
    with open(RES / "baselines.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(f"[{tag}] {name}: PR-AUC {m['pr_auc']:.4f}  ROC {m['roc_auc']:.4f}  "
          f"R@0.1% {m['rec@0.1%fpr']:.3f}  R@1% {m['rec@1%fpr']:.3f}  F1 {m['f1']:.3f}  "
          f"({m['secs']}s)  top: {m['top_feats'][:5]}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    canon = "--canonical-only" in a
    lf = float(a[a.index("--label-frac") + 1]) if "--label-frac" in a else 1.0
    sd = int(a[a.index("--seed") + 1]) if "--seed" in a else 0
    names = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--label-frac", "--seed"))]
    for n in names or ["sparkov", "paysim", "ibm_card", "aml_hi_small", "aml_li_small", "aml_hi_medium", "aml_li_medium"]:
        run(n, canon, lf, seed=sd)
