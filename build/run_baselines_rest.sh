#!/bin/sh
# Remaining Phase-2 baselines (2026-11-17), small first; big full runs last.
cd "$(dirname "$0")/.."
for ds in sparkov paysim ibm_card aml_hi_small aml_li_small; do python -u models/baseline_xgb.py $ds --canonical-only; done
for lf in 0.01 0.1; do for ds in sparkov paysim ibm_card aml_hi_small aml_li_small; do python -u models/baseline_xgb.py $ds --label-frac $lf; done; done
python -u models/baseline_xgb.py ibm_card
python -u models/baseline_xgb.py aml_li_medium
echo ALL_DONE
