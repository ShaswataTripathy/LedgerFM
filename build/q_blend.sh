#!/bin/sh
# Val-only blend weight selection, full labels, seed 0: FM fine-tune x plain XGBoost.
cd "$(dirname "$0")/.."
for ds in sparkov aml_li_small paysim ieee_cis ibm_card aml_hi_medium aml_li_medium; do
  python -u models/blend.py --ds $ds --fm-tag ${ds}__fm_all --xgb xgb
done
echo BLEND_DONE
