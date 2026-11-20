"""Score blend of a fine-tuned LedgerFM and an XGBoost model; the weight is chosen on val only.

Each score -> logit -> standardised with the VAL mean/std of that model (deployable: fixed constants),
blend = w * z_fm + (1 - w) * z_xgb, w in {0, .1, ..., 1} picked by val PR-AUC; F1 threshold from val.
Scores are cached in results/scores/<ds>/. Rows go to results/blend.jsonl (split=val, and split=test with --test).
Usage: python -u models/blend.py --ds aml_hi_small --fm-tag aml_hi_small__fm_all --xgb "xgb+emb[fm_all]" [--test]
"""
import argparse, json, pathlib, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "models"))
from ledgerfm.metrics import evaluate
from final_test import xgb_test, ft_test, latest, RES

SPLIT_ID = {"val": 1, "test": 2}


def scores(kind, r, split, device):
    d = RES / "scores" / r["ds"]; d.mkdir(parents=True, exist_ok=True)
    key = (r["tag"] if kind == "ft" else r["model"]).replace("[", "(").replace("]", ")")
    f = d / f"{key}__{split}.npz"
    if f.exists():
        z = np.load(f); return z["y"], z["s"], z["id"]
    y, s, i = xgb_test(r, split) if kind == "xgb" else ft_test(r, device, SPLIT_ID[split])
    o = np.argsort(i); y, s, i = y[o], s[o], i[o]
    np.savez(f, y=y, s=s, id=i)
    return y, s, i


def logit(s):
    s = np.clip(np.asarray(s, np.float64), 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--fm-tag", required=True)
    ap.add_argument("--xgb", required=True); ap.add_argument("--test", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    fr = latest(RES / "finetune.jsonl", lambda r: r["tag"])[a.fm_tag]
    xr = latest(RES / "baselines.jsonl", lambda r: (r["ds"], r["model"]))[(a.ds, a.xgb)]
    yv, fv, iv = scores("ft", fr, "val", a.device); yv2, xv, iv2 = scores("xgb", xr, "val", a.device)
    assert (iv == iv2).all() and (yv == yv2).all(), "val rows differ between models"
    lf, lx = logit(fv), logit(xv)
    cf, cx = (lf.mean(), lf.std()), (lx.mean(), lx.std())
    zf, zx = (lf - cf[0]) / cf[1], (lx - cx[0]) / cx[1]
    grid = [round(w, 1) for w in np.linspace(0, 1, 11)]
    res = {w: evaluate(yv, w * zf + (1 - w) * zx) for w in grid}
    w = max(grid, key=lambda g: res[g]["pr_auc"])
    curve = {str(g): round(res[g]["pr_auc"], 4) for g in grid}
    base = dict(ds=a.ds, fm_tag=a.fm_tag, xgb=a.xgb, w=w, curve_val=curve, norm=dict(fm=cf, xgb=cx))
    rows = [dict(res[w], split="val", **base)]
    print(f"[blend] {a.ds}: w_fm={w}  val PR-AUC {res[w]['pr_auc']:.4f} (FM {res[1.0]['pr_auc']:.4f}, "
          f"XGB {res[0.0]['pr_auc']:.4f})  curve {curve}", flush=True)
    if a.test:
        yt, ft, it = scores("ft", fr, "test", a.device); yt2, xt, it2 = scores("xgb", xr, "test", a.device)
        assert (it == it2).all()
        st = w * (logit(ft) - cf[0]) / cf[1] + (1 - w) * (logit(xt) - cx[0]) / cx[1]
        m = evaluate(yt, st, thr=res[w]["thr"])
        rows.append(dict(m, split="test", **base))
        print(f"[blend] {a.ds}: TEST PR-AUC {m['pr_auc']:.4f}", flush=True)
    with open(RES / "blend.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
