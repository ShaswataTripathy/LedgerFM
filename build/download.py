"""Download the core public Kaggle datasets for LedgerFM (no login needed).

Resumable: an archive that is already complete is skipped, and a partial
download continues via HTTP Range when the server supports it.
Usage: python -u build/download.py [name ...]
"""
import sys, time, zipfile, pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# name -> (kaggle dataset, optional single file inside it)
SOURCES = {
    "sparkov":   ("kartik2112/fraud-detection", None),
    "paysim":    ("ealaxi/paysim1", None),
    "ibm_card":  ("ealtman2019/credit-card-transactions", None),
    "elliptic":  ("ellipticco/elliptic-data-set", None),
    "aml_hi_small":  ("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "HI-Small_Trans.csv"),
    "aml_li_small":  ("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "LI-Small_Trans.csv"),
    "aml_hi_medium": ("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "HI-Medium_Trans.csv"),
    "aml_li_medium": ("ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "LI-Medium_Trans.csv"),
}
API = "https://www.kaggle.com/api/v1/datasets/download/"
UA = {"User-Agent": "ledgerfm-downloader/0.1"}


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers=dict(UA, **({"Range": f"bytes={have}-"} if have else {})))
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored Range: restart
            have = 0
        total = int(r.headers.get("Content-Length") or 0) + have
        mode = "ab" if have else "wb"
        t0, done, last = time.time(), have, 0
        with open(tmp, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 15:
                    last = time.time()
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    eta = (total - done) / max(rate * 1e6, 1) / 60 if total else 0
                    print(f"  {dest.name}: {done/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s  eta {eta:.0f} min", flush=True)
    tmp.rename(dest)


def main(names):
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names:
        ds, file = SOURCES[name]
        out = RAW / name
        if (out / ".done").exists():
            print(f"[skip] {name}", flush=True)
            continue
        out.mkdir(exist_ok=True)
        url = API + ds + (f"/{file}" if file else "")
        z = out / f"{name}.zip"
        print(f"[get ] {name} <- {ds}{'/' + file if file else ''}", flush=True)
        for attempt in range(5):
            try:
                if not z.exists():
                    fetch(url, z)
                break
            except Exception as e:
                print(f"  retry {attempt+1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"[FAIL] {name}", flush=True)
            continue
        if zipfile.is_zipfile(z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            z.unlink()
        else:  # single files may arrive uncompressed
            z.rename(out / (file or f"{name}.bin"))
        (out / ".done").touch()
        print(f"[done] {name}: {[p.name for p in out.iterdir()]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SOURCES))
