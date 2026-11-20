# ledgerfm

An open foundation model for **transaction sequences**, for fraud detection and anti-money-laundering (AML).

LedgerFM reads an account's transaction history and returns a 256-dimensional embedding for every transaction. Use the embeddings as extra features, fine-tune the model, or blend its scores with the XGBoost model you already run.

- Weights (Apache-2.0, 26.6M parameters) on Hugging Face: https://huggingface.co/tripathyShaswata/LedgerFM
- Code, training pipeline and every result: https://github.com/ShaswataTripathy/LedgerFM

```bash
pip install ledgerfm
```

## Embed transactions

```python
import polars as pl
from ledgerfm import load, embed

fm = load()                      # downloads the v0.1 weights from Hugging Face once
df = pl.read_parquet("transactions.parquet")
E = embed(fm, df)                # txn_id, emb0 .. emb255
```

Or from the command line, reducing to 32 dimensions for use as XGBoost features:

```bash
ledgerfm embed transactions.parquet embeddings.parquet --pca 32
```

### Input columns

| column | type | |
|---|---|---|
| `txn_id` | int | unique id |
| `ts` | datetime | transaction time |
| `account` | str | sender / card / customer (the model reads this account's history) |
| `counterparty` | str | receiver / merchant |
| `amount`, `currency` | float, str | |
| `amount_recv`, `currency_recv`, `category`, `channel`, `location`, `instrument`, `src_bank`, `dst_bank`, `errors` | optional | may be null |

The tokenizer uses no dataset statistics, so you don't need to refit anything for your own data.

## Blend with your existing model

```python
from ledgerfm import Blend
b = Blend().fit(y_val, p_ledgerfm_val, p_xgb_val)   # picks the weight on validation
score = b.predict(p_ledgerfm_new, p_xgb_new)
print(b.w, b.curve)
```

If LedgerFM adds nothing on your data, the blend picks weight 0 and you keep your XGBoost result.

## How good is it?

On a simulated banking world the model never saw (IBM AML HI-Large, 22.6M transactions), test PR-AUC with full labels:

| model | test PR-AUC |
|---|---|
| XGBoost | 0.289 |
| LedgerFM v0.1, fine-tuned | 0.332 |
| blend of the two | 0.338 |

Limits: it was trained mostly on synthetic data; XGBoost wins on short account histories and with few labels. Full results and limitations are in the [GitHub README](https://github.com/ShaswataTripathy/LedgerFM).
