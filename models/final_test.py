"""ONE-TIME test-set evaluation. Run only after every model choice is frozen on val.

Scores split=test for every model recorded in results/baselines.jsonl (XGBoost, incl. xgb+emb) and
results/finetune.jsonl (LedgerFM / scratch fine-tunes). F1 uses the threshold chosen on val, never on test.
Writes results/final_test.jsonl and refuses to touch a run that already has a test result.
Run key: XGBoost rows -> (ds, model); fine-tune rows -> their unique tag (so seed reruns `_s<k>` are separate runs).

Usage: python -u models/final_test.py --confirm [--ds aml_hi_small,sparkov] [--dry-run]
"""
import argparse, json, pathlib, sys, time
import numpy as np, polars as pl, torch, xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models"))
from ledgerfm.data import TokenStore
from ledgerfm.model import LedgerFM, Config
from ledgerfm.infer import run_rows
from ledgerfm.metrics import evaluate
from baseline_xgb import FEAT, to_pandas
from finetune import Scorer

RES = ROOT / "results"; OUT = RES / "final_test.jsonl"


def latest(path, key):
    rows = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); rows[key(r)] = r  # last run of a config wins
    return rows


def xgb_test(r, split="test"):
    ds, tag = r["ds"], r["model"]
    bst = xgb.Booster(); bst.load_model(RES / "baselines" / f"{ds}_{tag}.ubj")
    df = pl.read_parquet(FEAT / f"{ds}.parquet")
    if "+emb[" in tag:
        emb = pl.read_parquet(ROOT / "data/emb" / tag.split("[")[1].split("]")[0] / f"{ds}.parquet")
        df = df.join(emb.with_columns(pl.col("txn_id").cast(df["txn_id"].dtype)), on="txn_id", how="left")
    feats = bst.feature_names
    cats = {c: df[c].cast(pl.String).drop_nulls().unique().sort().to_list()
            for c in feats if df[c].dtype == pl.Categorical}
    te = df.filter(pl.col("split") == split)
    d = xgb.DMatrix(to_pandas(te, feats, cats), enable_categorical=True)
    return te["label"].to_numpy(), bst.predict(d, iteration_range=(0, r["best_iter"] + 1)), te["txn_id"].to_numpy()


def ft_test(r, device, split=2):
    ck = torch.load(ROOT / "checkpoints/ft" / f"{r['tag']}.pt", map_location="cpu", weights_only=False)
    model = Scorer(LedgerFM(Config(**ck["cfg"]))); model.load_state_dict(ck["state"]); model.to(device).eval()
    st = TokenStore(r["ds"])
    out = np.full(st.N, np.nan, np.float32)

    def fn(h, idx):
        out[idx] = torch.sigmoid(model.head(h)).squeeze(-1).float().cpu().numpy()
    run_rows(model.fm, st, model.fm.c.max_len, {split}, fn, device=device)
    m = np.asarray(st.split) == split
    assert not np.isnan(out[m]).any(), "some test rows were not scored"
    return np.asarray(st.label)[m], out[m], np.asarray(st.txn_id)[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="required: acknowledges the test set is used once")
    ap.add_argument("--ds", default=""); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    only = set(filter(None, a.ds.split(",")))
    done = latest(OUT, lambda r: r["key"])
    todo = []
    for r in latest(RES / "baselines.jsonl", lambda r: (r["ds"], r["model"])).values():
        todo.append(("xgb", r, f"{r['ds']}::{r['model']}"))
    for r in latest(RES / "finetune.jsonl", lambda r: r["tag"]).values():
        todo.append(("ft", r, r["tag"]))
    todo = [t for t in todo if (not only or t[1]["ds"] in only) and t[2] not in done]
    print(f"{len(todo)} runs to test; {len(done)} already tested (skipped)")
    for kind, r, k in todo:
        print(f"  {kind:3s} {k}")
    if a.dry_run or not todo:
        return
    if not a.confirm:
        sys.exit("refusing: pass --confirm (the test split is evaluated exactly once)")
    for kind, r, k in todo:
        t0 = time.time()
        y, s, _ = xgb_test(r) if kind == "xgb" else ft_test(r, a.device)
        m = evaluate(y, s, thr=r["thr"])
        m.update(key=k, ds=r["ds"], model=r["model"], tag=r.get("tag", r["model"]), label_frac=r.get("label_frac"),
                 seed=r.get("seed", 0), split="test", val_pr_auc=r["pr_auc"],
                 secs=round(time.time() - t0))
        with open(OUT, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"[test] {k}: PR-AUC {m['pr_auc']:.4f} (val {r['pr_auc']:.4f})  ROC {m['roc_auc']:.4f}  "
              f"R@1% {m['rec@1%fpr']:.3f}  F1@valthr {m['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
