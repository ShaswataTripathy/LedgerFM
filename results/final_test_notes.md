# Final test (run once, 2026-11-20)

142 frozen validation runs scored on split=test (`results/final_test.jsonl`, log `logs/final_test.log`).
Thresholds for F1 come from val. Nothing was tuned after this run.

## ibm_card: real distribution shift (not a bug)
Every model collapses from val to test (FM .940 -> .191, xgb .836 -> .026). Fraud base rate is similar
(train .12%, val .14%, test .10%), but the fraud *type* changes over time:

| split | dates | fraud by channel |
|---|---|---|
| train | 1991-2014 | Online 12,371 / Swipe 5,693 / Chip 0 |
| val | 2014-2017 | Online 5,850 / Chip 580 / Swipe 430 |
| test | 2017-2020 | Chip 4,256 / Swipe 449 / Online 128 |

88% of test fraud is chip-card fraud; training has none (EMV chip roll-out in the simulator).
Kept as a reported finding: the pretrained FM degrades least (.191 vs .026-.036 for XGBoost, 7x).

# Fresh test: IBM AML HI-Large, 2022-08-01..08-11 (run once, 2026-11-20 evening)
Never touched before; own 60/20/20 temporal split (test: 4.53M txns, 3,095 laundering). All choices made on its val.

| model (full labels) | val PR-AUC | test PR-AUC |
|---|---|---|
| XGBoost | .268 | .289 |
| LedgerFM no pretraining | .172 | .202 |
| LedgerFM fm_all (train-period pretrain) | .291 | .305 |
| LedgerFM fm_v01 (release, train+val pretrain) | .300 | .332 |
| Blend fm_v01 x XGBoost (w_fm .9, val-chosen) | .315 | **.338** |

10% labels: XGBoost .246, fm_v01 .199, blend (val chose w_fm 0 = XGBoost) .246.
