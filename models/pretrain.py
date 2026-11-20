"""Self-supervised pretraining of LedgerFM (masked fields + next transaction). No labels are used.

Uses only split=train rows of the pretraining datasets; val-period rows give the pretraining val loss.
Resumable: rerun the same command to continue from checkpoints/<run>/last.pt.
Usage:
  python -u models/pretrain.py --run fm_all --sets aml_hi_small,aml_li_small,aml_hi_medium,aml_li_medium,ibm_card,sparkov
  options: --steps 30000 --bs 64 --L 128 --d 256 --layers 6 --lr 6e-4 --temp 0.5
"""
import argparse, json, math, pathlib, queue, sys, threading, time
import numpy as np, torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ledgerfm.data import TokenStore, pack, make_batch
from ledgerfm.model import LedgerFM, Config, n_params


class Stream:
    """Endless packed batches from one dataset's chunks in [lo, hi] split range."""
    def __init__(self, store, L, bs, seed, lo=0, hi=0):
        self.st, self.L, self.bs, self.lo, self.hi = store, L, bs, lo, hi
        self.rng = np.random.default_rng(seed); self.epoch = 0; self._new_epoch()
        self.lock = threading.Lock()  # several producer threads share one Stream

    def _new_epoch(self):
        self.s, self.l = self.st.train_chunks(self.L, self.rng, self.lo, self.hi)
        self.rows = pack(self.l, self.L, self.rng.permutation(len(self.s)))
        self.i = 0; self.epoch += 1

    def tokens(self):
        return int(self.l.sum())

    def next(self):
        with self.lock:  # snapshot rows + chunk arrays together; _new_epoch swaps all three
            if self.i + self.bs > len(self.rows):
                self._new_epoch()
            r = self.rows[self.i:self.i + self.bs]; self.i += self.bs
            s, l = self.s, self.l
        return make_batch(self.st, s, l, r, self.L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--sets", required=True)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--L", type=int, default=128); ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-hi", type=int, default=0, help="1 = also pretrain on val-period rows (release weights; val loss then not held out, use last.pt)")
    a = ap.parse_args()
    out = ROOT / "checkpoints" / a.run; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    sets = a.sets.split(",")
    stores = {n: TokenStore(n) for n in sets}
    train = {n: Stream(stores[n], a.L, a.bs, seed=i, hi=a.train_hi) for i, n in enumerate(sets)}
    sizes = np.array([train[n].tokens() for n in sets], float)
    w = sizes ** a.temp; w /= w.sum()
    print("mixture:", {n: f"{t/1e6:.1f}M tok, p={p:.2f}" for n, t, p in zip(sets, sizes, w)}, flush=True)
    # fixed val batches (val-period rows; context limited to the val period, which is conservative)
    val = {n: [Stream(stores[n], a.L, a.bs, seed=100 + i, lo=1, hi=1).next() for _ in range(8)] for i, n in enumerate(sets)}

    cfg = Config(d=a.d, layers=a.layers, heads=a.heads, max_len=a.L)
    model = LedgerFM(cfg).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=dev.type == "cuda")
    step, best = 0, float("inf")
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step, best = s["step"], s["best"]
        print(f"resumed at step {step}", flush=True)
    print(f"params {n_params(model)/1e6:.1f}M  cfg {cfg.to_dict()}", flush=True)
    json.dump(dict(vars(a), cfg=cfg.to_dict(), mixture=dict(zip(sets, w.tolist()))), open(out / "args.json", "w"), indent=1)

    # background batch producer
    q = queue.Queue(maxsize=8)
    def producer(k):
        rng = np.random.default_rng(step * 10 + k)  # Generator is not thread-safe: one per thread
        while True:
            n = sets[rng.choice(len(sets), p=w)]
            q.put((n, train[n].next()))
    for k in range(3):
        threading.Thread(target=producer, args=(k,), daemon=True).start()

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup)))))

    @torch.no_grad()
    def evaluate():
        model.eval(); res = {}
        torch.manual_seed(0)
        for n, bl in val.items():
            ls = []
            for b in bl:
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    l, _ = model.pretrain_loss(*[t.to(dev) for t in b[:4]])
                ls.append(float(l))
            res[n] = float(np.mean(ls))
        model.train(); return res

    model.train(); t0 = time.time(); tok = 0; run_loss = None; step0 = step
    while step < a.steps:
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step)
        n, b = q.get()
        ints, fl, seg, pos = [t.to(dev, non_blocking=True) for t in b[:4]]
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            loss, parts = model.pretrain_loss(ints, fl, seg, pos)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; tok += int((seg > 0).sum())
        run_loss = float(loss) if run_loss is None else 0.98 * run_loss + 0.02 * float(loss)
        if step % 100 == 0:
            el = time.time() - t0
            done = step - step0
            print(f"step {step}/{a.steps}  loss {run_loss:.3f}  lr {lr_at(step):.2e}  {tok/el/1e3:.0f}k tok/s  "
                  f"eta {(a.steps - step) * el / max(done, 1) / 60:.0f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(); vm = float(np.mean(list(v.values())))
            print(f"[val] step {step}  mean {vm:.4f}  " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()), flush=True)
            state = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=min(best, vm), cfg=cfg.to_dict())
            torch.save(state, out / "last.pt")
            if vm < best:
                best = vm; torch.save(dict(model=model.state_dict(), step=step, cfg=cfg.to_dict(), val=v), out / "best.pt")
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, val=v, train=run_loss, secs=time.time() - t0)) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
