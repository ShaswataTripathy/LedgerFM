"""Push LedgerFM to the Hugging Face Hub: weights, code, results, README (model card).
Usage: python push_to_hub.py --repo_id tripathyShaswata/LedgerFM"""
import argparse, pathlib, shutil, tempfile
from huggingface_hub import HfApi, create_repo

ROOT = pathlib.Path(__file__).resolve().parent
CARD = """---
license: apache-2.0
library_name: pytorch
tags: [fraud-detection, anti-money-laundering, aml, transactions, foundation-model, tabular, time-series, finance]
datasets: [ealtman2019/ibm-transactions-for-anti-money-laundering-aml]
pipeline_tag: feature-extraction
---

"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repo_id", default="tripathyShaswata/LedgerFM")
    a = ap.parse_args()
    create_repo(a.repo_id, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        for p in ["ledgerfm", "models", "build", "results", "docs", "examples", "notebooks", "demo"]:
            shutil.copytree(ROOT / p, t / p, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in ["LICENSE", "requirements.txt", "push_to_hub.py", "pyproject.toml", "PYPI.md"]:
            shutil.copy(ROOT / f, t / f)
        for f in ["model.safetensors", "config.json"]:
            shutil.copy(ROOT / "weights" / f, t / f)
        (t / "README.md").write_text(CARD + (ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
        HfApi().upload_folder(folder_path=t, repo_id=a.repo_id, repo_type="model",
                              commit_message="LedgerFM v0.1")
    print("done:", f"https://huggingface.co/{a.repo_id}")
