#!/bin/sh
# Full Phase-2 baseline grid (val only). Resumable by rerunning a missing line.
cd "$(dirname "$0")/.."
for ds in paysim ibm_card aml_li_small aml_hi_medium aml_li_medium; do python -u models/baseline_xgb.py $ds; done
for ds in sparkov paysim ibm_card aml_hi_small aml_li_small; do python -u models/baseline_xgb.py $ds --canonical-only; done
for lf in 0.01 0.1; do for ds in sparkov paysim ibm_card aml_hi_small aml_li_small; do python -u models/baseline_xgb.py $ds --label-frac $lf; done; done
echo ALL_DONE
python -u models/baseline_xgb.py ibm_card
