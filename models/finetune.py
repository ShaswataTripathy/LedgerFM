"""Fine-tune LedgerFM (or a same-size scratch model) on one dataset's labels. Val metrics only.

Usage:
  python -u models/finetune.py --ds aml_hi_small --init checkpoints/fm_all/best.pt [--label-frac 0.01]
  python -u models/finetune.py --ds aml_hi_small --init scratch          # no-pretraining baseline
The best state (by val PR-AUC) is saved for the one-time test run.
"""
import argparse, json, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ledgerfm.data import TokenStore, pack, make_batch
from ledgerfm.model import LedgerFM, Config
from ledgerfm.infer import run_rows
from ledgerfm.metrics import evaluate, label_keep


class Scorer(nn.Module):
    def __init__(self, fm):
        super().__init__()
        self.fm = fm
        d = fm.c.d
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, ints, fl, seg, pos):
        return self.head(self.fm.encode(ints, fl, seg, pos)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--label-frac", type=float, default=1.0); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="training + label-subset seed; >0 adds _s<seed> to the tag")
    ap.add_argument("--freeze", action="store_true", help="linear-probe style: train only the head")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.init == "scratch":
        cfg = Config(**json.load(open(ROOT / "checkpoints/fm_all/args.json"))["cfg"]) \
            if (ROOT / "checkpoints/fm_all/args.json").exists() else Config()
        fm = LedgerFM(cfg); init_name = "scratch"
    else:
        s = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        fm = LedgerFM(Config(**s["cfg"])); fm.load_state_dict(s["model"]); init_name = pathlib.Path(a.init).parent.name
    L = fm.c.max_len
    model = Scorer(fm).to(dev)
    if a.freeze:
        for p in model.fm.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": [p for p in model.fm.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    st = TokenStore(a.ds)
    lab = np.asarray(st.label); keep = label_keep(np.asarray(st.txn_id), a.label_frac, a.seed)
    s0, l0 = st.train_chunks(L, rng, 0, 0)
    tr_pos = int((lab[(np.asarray(st.split) == 0)] * keep[np.asarray(st.split) == 0]).sum())
    tr_n = int(keep[np.asarray(st.split) == 0].sum())
    pw = float(min(100.0, np.sqrt((tr_n - tr_pos) / max(tr_pos, 1))))
    tag = a.tag or f"{a.ds}__{init_name}{'_frz' if a.freeze else ''}{'' if a.label_frac >= 1 else f'_lf{a.label_frac}'}{f'_s{a.seed}' if a.seed else ''}"
    print(f"[ft] {tag}: train labels {tr_n:,} (pos {tr_pos:,}), pos_weight {pw:.1f}", flush=True)

    # oversample chunks that contain kept positives so rare-label runs still see them
    cum = np.r_[0, np.cumsum(lab * keep)]
    has_pos = (cum[s0 + l0] - cum[s0]) > 0
    p_chunk = np.where(has_pos, 10.0, 1.0); p_chunk /= p_chunk.sum()

    def val_scores():
        out = np.full(st.N, np.nan, np.float32)
        def fn(h, idx):
            out[idx] = torch.sigmoid(model.head(h)).squeeze(-1).cpu().numpy()
        run_rows(model.fm, st, L, {1}, fn, device=dev)
        model.train(); m = np.asarray(st.split) == 1
        return out[m], lab[m]

    best, best_state, bad, t0 = -1, None, 0, time.time()
    hist = []
    model.train()
    for step in range(1, a.steps + 1):
        ch = rng.choice(len(s0), size=a.bs * 4, p=p_chunk)
        rows = pack(l0, L, ch)[:a.bs]
        ints, fl, seg, pos, idx = make_batch(st, s0, l0, rows, L)
        idx_np = idx.numpy(); ok = idx_np >= 0
        y = np.zeros(idx_np.shape, np.float32); w = np.zeros(idx_np.shape, np.float32)
        y[ok] = lab[idx_np[ok]]; w[ok] = keep[idx_np[ok]]
        with torch.autocast(dev, dtype=torch.bfloat16):
            logit = model(ints.to(dev), fl.to(dev), seg.to(dev), pos.to(dev)).float()
        yt, wt = torch.from_numpy(y).to(dev), torch.from_numpy(w).to(dev)
        loss = (F.binary_cross_entropy_with_logits(logit, yt, reduction="none",
                pos_weight=torch.tensor(pw, device=dev)) * wt).sum() / wt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            print(f"  step {step}/{a.steps} loss {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            s, yv = val_scores(); m = evaluate(yv, s)
            hist.append(dict(step=step, pr_auc=m["pr_auc"]))
            print(f"  [val] step {step} PR-AUC {m['pr_auc']:.4f} ROC {m['roc_auc']:.4f} F1 {m['f1']:.3f}", flush=True)
            if m["pr_auc"] > best:
                best, bad, best_m = m["pr_auc"], 0, m
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
    out = ROOT / "checkpoints/ft"; out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, cfg=fm.c.to_dict(), args=vars(a)), out / f"{tag}.pt")
    best_m.update(ds=a.ds, model=f"fm:{init_name}{'_frz' if a.freeze else ''}", label_frac=a.label_frac,
                  split="val", seed=a.seed, train_pos=tr_pos, secs=round(time.time() - t0), hist=hist, tag=tag)
    with open(ROOT / "results/finetune.jsonl", "a") as f:
        f.write(json.dumps(best_m) + "\n")
    print(f"[ft] {tag}: best val PR-AUC {best:.4f}  ROC {best_m['roc_auc']:.4f}  R@1% {best_m['rec@1%fpr']:.3f}  "
          f"F1 {best_m['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
