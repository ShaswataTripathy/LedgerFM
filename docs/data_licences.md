# LedgerFM: data licences (Phase 0)

Checked 2026-11-16 at the primary sources: the Kaggle API dataset metadata, the GitHub API licence fields and the SPDX licence texts.

**Target:** release the weights under Apache-2.0. A dataset is used to train the released weights only if its licence clearly allows publishing a model trained on it.

## Final dataset list

| Dataset | Source | Licence (verified) | Size (CSV) | Role |
|---|---|---|---|---|
| IBM AMLworld HI-Small, LI-Small, HI-Medium, LI-Medium | kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | **CDLA-Sharing-1.0** | 0.48 + 0.65 + 3.03 + 2.98 GB | **Pretrain + AML eval** |
| IBM AMLworld HI-Large / LI-Large | same | CDLA-Sharing-1.0 | 17.1 / 16.7 GB | Optional later (disk) |
| IBM credit-card transactions (TabFormer data) | kaggle `ealtman2019/credit-card-transactions`; github `IBM/TabFormer` | **Apache-2.0** (full text in the Kaggle description; repo is also Apache-2.0) | 2.35 GB, 24M rows | **Pretrain + fraud eval** |
| Sparkov | kaggle `kartik2112/fraud-detection` | **CC0** | 0.50 GB, 1.85M rows | **Pretrain + fraud eval** |
| PaySim | kaggle `ealaxi/paysim1` | CC BY-SA 4.0 | 0.49 GB, 6.3M rows | **Eval only** (unseen-domain transfer) |
| Elliptic (v1) | kaggle `ellipticco/elliptic-data-set` | CC BY-NC-ND 4.0 | 0.70 GB | **Eval only** (real Bitcoin data, research use) |
| Feedzai BAF | kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022` | CC BY-NC-SA 4.0 | 1.36 GB | Eval only, optional (tabular, not sequences) |
| Elliptic2 | kaggle `ellipticco/elliptic2-data-set` | CC BY-NC-ND 4.0 | **88 GB** | **Dropped** (does not fit on disk) |
| IEEE-CIS (Kaggle competition) | competition rules | Rules page needs login; Kaggle competition data is normally non-commercial / research use only | 1.3 GB | **Deferred**: needs your Kaggle login and rule acceptance; eval only if used |

## Why each decision

- **CDLA-Sharing-1.0 (IBM AML).** §1.11 defines "Results" as "the outcomes or outputs that You obtain from Your Computational Use of Data", and §3.5 says: *"This Agreement imposes no obligations or restrictions on Your Use or Publication of Results."* Trained weights are Results as long as they don't memorise more than a de minimis portion of the data. The share-alike part covers only republishing the data itself, which we won't do. → OK to train on.
- **Apache-2.0 (IBM card data)** and **CC0 (Sparkov)** → OK to train on.
- **PaySim (CC BY-SA 4.0).** Whether trained weights count as an "adaptation" that must stay share-alike is legally unsettled. Keeping PaySim out of training avoids the question. It also gives a clean transfer target, because it is a domain the model has never seen (mobile money).
- **Elliptic, BAF (NC licences).** Evaluation only. We publish benchmark numbers, never the data or weights trained on it.
- **NVIDIA baseline.** The NVIDIA AI Blueprint `financial-fraud-detection` code is Apache-2.0. Its training container (GNN + XGBoost) needs an NGC account and runs under the NVIDIA Software License Agreement. Hugging Face has **no** NVIDIA transaction/fraud checkpoint (API search for author=nvidia, 2026-11-16). → As the baseline we reimplement the open TabFormer/TabBERT recipe from `IBM/TabFormer` (Apache-2.0) and don't depend on the NGC container.
- **Name.** "LedgerFM" is free on Hugging Face (models and datasets), GitHub repos and PyPI (checked 2026-11-16). Note: Ledger SAS is a hardware-wallet company. Same word, different field, so this is low risk.

## Resulting licence for the weights
The training data is IBM AML (CDLA-Sharing, Results unrestricted), IBM card (Apache-2.0) and Sparkov (CC0). → **The weights can be released under Apache-2.0.** The model card will cite IBM, Sparkov and the evaluation datasets.

## Disk budget (60 GB free on D:)
Phase 1 core downloads come to about 10.5 GB of CSV, which becomes roughly 3 GB of Parquet once the CSVs are deleted. Downloading one Large AML set (about 17 GB of CSV, converted in chunks) still fits later if needed.

## Leave-one-dataset-out transfer matrix (planned)
- Pretrain sources: AML-Small/Medium, IBM card, Sparkov.
- Transfer targets: PaySim (never pretrained on), Elliptic (real data, never pretrained on), and each pretrain source held out in turn.
