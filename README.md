# LedgerFM v0.1

**An open foundation model for transaction sequences, for fraud and anti-money-laundering (AML) work.**

Large payment companies run transaction foundation models, but none of them are open (Stripe, Visa, Mastercard, Feedzai and Revolut all keep theirs closed). LedgerFM is a small open alternative. It is a 26.6M-parameter causal transformer that reads an account's transaction history and returns one embedding per transaction. You can fine-tune it, or add its embeddings to the XGBoost features you already use.

- Weights: Apache-2.0, trained only on permissively licensed synthetic data (see [Training data](#training-data)).
- Recommended use: **fine-tuned LedgerFM blended with your XGBoost model**, with the blend weight chosen on your own validation data.
- Honest scope: it helps on **long account histories**. On short histories, or when you have very few labels, XGBoost alone is as good or better. Details below.

## Results

PR-AUC is the main metric (positives are 0.05–1% of rows). Every dataset uses a 60/20/20 **temporal** split. All model choices were made on validation, and each test set was scored once.

### Fresh test: a simulated world no model had seen

IBM AML **HI-Large**, 2022-08-01 to 08-11: 22.6M transactions, 3,095 laundering cases in the test period. LedgerFM was never pretrained on this world, and its weights were frozen before this data was downloaded.

| Model (full labels) | Val | **Test** |
|---|---|---|
| XGBoost (29 causal features: windows, sender and receiver history) | .268 | .289 |
| LedgerFM architecture, no pretraining | .172 | .202 |
| LedgerFM pretrained on train period only (`fm_all`, research run) | .291 | .305 |
| **LedgerFM v0.1** (released weights), fine-tuned | .300 | **.332** |
| **LedgerFM v0.1 + XGBoost blend** (w_fm = 0.9, chosen on val) | .315 | **.338** |

With **10% of the labels**, XGBoost is better: .246 against .199 for LedgerFM. The validation-chosen blend put all its weight on XGBoost, giving .246.

### Benchmark study (research weights `fm_all`, test PR-AUC)

| Dataset | History | LedgerFM fine-tuned | No pretraining | XGBoost | XGBoost + LedgerFM emb. |
|---|---|---|---|---|---|
| Sparkov (card fraud) | long | **.991** | .942 | .946 | .956 |
| IBM AML HI-Medium | long | **.764** | .672 | .726 | .747 |
| IBM AML LI-Medium | long | **.369** | .214 | .357 | .346 |
| IBM AML HI-Small (3 seeds) | long | .696 ± .005 | .479 | .686 ± .004 | **.707 ± .006** |
| IBM AML LI-Small (3 seeds) | long | **.313 ± .010** | .070 | .303 ± .011 | .310 ± .005 |
| IEEE-CIS (real, eval only) | short | .152 | .118 | **.492** | .487 |
| PaySim (eval only, balance fields removed) | short | **.509** | .415 | .460 | .406 |
| IBM card (TabFormer data) | long | **.191** | .032 | .026 | .036 |

**Few labels on Sparkov** (test): with 10% of the labels, LedgerFM scores .963 against .885 for XGBoost. With 1%, it is .787 against .621.

**Blending on validation:** the blend was at least as good as the better single model on every dataset. On IEEE-CIS it reached .626, against .604 for XGBoost and .269 for LedgerFM. On IBM AML HI-Small it reached .626, against .597 and .595.

**Leave-one-dataset-out (transfer):**
- Sparkov: LedgerFM pretrained *without* Sparkov still scores .978 on test (.799 with 1% labels), against .946 (.621) for XGBoost.
- AML LI-Small: pretrained without either AML LI world, it scores .310 ± .010, tying XGBoost (.303). With 10% labels it trails XGBoost.

## Limitations

- **Mostly synthetic data.** Training and most evaluation use simulators (IBM AMLworld, IBM card, Sparkov). The one real dataset (IEEE-CIS) has short histories, and there LedgerFM alone is weak. Validate on your own data before relying on it.
- **Short histories.** If most accounts have only a few transactions, use XGBoost, or the blend, which chooses this for you on validation.
- **Few labels.** Except on Sparkov, the low-label advantage over XGBoost did not hold up. On the fresh test with 10% labels, XGBoost won.
- **Drift.** In the IBM card data, 88% of test-period fraud is chip-card fraud, and the training years contain none. Every model collapses on test (LedgerFM .191, XGBoost .026; validation was about .9). See `results/final_test_notes.md`.
- **Sender history only.** The model sees the sender's past plus summary statistics of the receiver. A two-sided variant that encodes the receiver's full history was tried and did not improve validation (`models/twoside_xgb.py`), so it is not part of v0.1.
- PaySim with its raw balance fields is trivially separable (XGBoost scores 1.000). The PaySim row above removes those fields.

## Input format

One row per transaction (a polars DataFrame). Missing optional fields can be null.

| Column | Type | Notes |
|---|---|---|
| `txn_id` | int64 | unique |
| `ts` | datetime | transaction time |
| `account` | str | sender / card / customer. The model reads this account's history |
| `counterparty` | str | receiver / merchant |
| `amount`, `currency` | float, str | converted to USD on a fixed scale |
| `amount_recv`, `currency_recv` | float, str | optional (cross-currency) |
| `category`, `channel`, `location`, `instrument`, `src_bank`, `dst_bank`, `errors` | str | optional, hashed |

The tokenizer uses no dataset statistics (fixed transforms and md5 hashing), so new institutions need no refitting.

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShaswataTripathy/LedgerFM/blob/main/notebooks/LedgerFM_quickstart.ipynb): a 5-minute walkthrough on 1.85M public card transactions ([demo dataset](https://huggingface.co/datasets/tripathyShaswata/ledgerfm-demo-sparkov)). On test PR-AUC, XGBoost on per-transaction fields scores 0.796; adding LedgerFM embeddings brings it to 0.943.

```bash
pip install ledgerfm
```

```python
from ledgerfm import load, embed, Blend
fm = load()                              # tripathyShaswata/LedgerFM, 106 MB safetensors
E = embed(fm, df)                        # polars: txn_id, emb0..emb255

b = Blend().fit(y_val, p_fm_val, p_xgb_val)   # weight chosen on validation
score = b.predict(p_fm_new, p_xgb_new)
```

```bash
ledgerfm embed transactions.parquet embeddings.parquet --pca 32   # command line
```

To work from the repo instead: `pip install -r requirements.txt` and `python examples/quickstart.py`.

Recommended recipe, as used for the results above:
1. Fine-tune with `models/finetune.py` (sequence classifier: the encoder plus a linear head, 6k steps). Separately, train XGBoost on your tabular features.
2. Blend the two with `models/blend.py`: standardised logits, `w * z_fm + (1 - w) * z_xgb`, with `w` picked on validation.
3. Cheaper alternative: reduce the frozen embeddings to 32 dimensions with PCA and add them to XGBoost (`models/embed_xgb.py`). This gives a small, consistent gain with full labels on AML; it did not help with few labels.

## Model

- **Encoder:** 6 layers, d = 256, 8 heads, context of 128 transactions. Each transaction is the sum of 14 categorical field embeddings and 12 continuous features.
- **Packing:** several accounts are packed into one sequence, with causal attention within each account.
- **Pretraining objectives:** masked fields (amount bucket, category, channel, location, currency, counterparty, hour) and next-transaction prediction (amount, category, channel, time gap, flags).
- **v0.1 training run:** 25k steps, batch 128, on the train and validation periods of the pretraining sets. The test periods were never used. One RTX 4070 Laptop GPU (8 GB), about 50 minutes.

## Training data

| Dataset | Licence | Role |
|---|---|---|
| IBM AMLworld HI/LI-Small, HI/LI-Medium | CDLA-Sharing-1.0 (§3.5: no restrictions on Results) | pretraining + evaluation |
| IBM credit-card transactions (TabFormer) | Apache-2.0 | pretraining + evaluation |
| Sparkov | CC0 | pretraining + evaluation |
| IBM AMLworld HI-Large (11-day slice) | CDLA-Sharing-1.0 | fresh test only |
| PaySim (CC BY-SA 4.0), IEEE-CIS (competition rules) | | evaluation only |

The licence reasoning is in `docs/data_licences.md`. No data is redistributed here.

## Reproducing

The `build/` scripts download the data, convert it to the canonical schema, then build features and tokens. The `models/` scripts cover pretraining, fine-tuning, the baselines, blending and the one-time test. The `build/q_*.sh` scripts are the exact run queues. Every number above is in `results/*.jsonl`.

## Citation

```
@misc{ledgerfm2026,
  title  = {LedgerFM: an open foundation model for transaction sequences},
  author = {Tripathy, Shaswata},
  year   = {2026},
  url    = {https://huggingface.co/tripathyShaswata/LedgerFM}
}
```
