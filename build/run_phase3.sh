#!/bin/sh
# Phase 3 (val only): LedgerFM fine-tune vs same-size scratch vs XGB(+FM embeddings), full and low-label.
# Each line is independent; rerun a missing one. GPU jobs are sequential.
cd "$(dirname "$0")/.."
FM=checkpoints/fm_all/best.pt
SETS="sparkov paysim aml_hi_small aml_li_small ibm_card aml_hi_medium aml_li_medium"
for lf in 1.0 0.1 0.01; do
  for ds in $SETS; do
    python -u models/finetune.py --ds $ds --init $FM --label-frac $lf
    python -u models/finetune.py --ds $ds --init scratch --label-frac $lf
    python -u models/embed_xgb.py --ds $ds --init $FM --label-frac $lf
  done
done
# linear probe (frozen encoder) on full labels
for ds in $SETS; do python -u models/finetune.py --ds $ds --init $FM --freeze; done
echo PHASE3_DONE
