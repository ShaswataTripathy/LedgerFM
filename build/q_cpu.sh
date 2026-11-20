#!/bin/sh
# CPU queue: XGBoost + frozen LedgerFM embeddings. Skips (ds, model) already in results/baselines.jsonl.
cd "$(dirname "$0")/.."
FM=checkpoints/fm_all/best.pt
SETS="sparkov paysim aml_hi_small aml_li_small ibm_card aml_hi_medium aml_li_medium ieee_cis"
done_m() { grep -F "\"ds\": \"$1\", \"model\": \"$2\"," results/baselines.jsonl >/dev/null 2>&1; }
# wait for the ibm_card run that was started by the old sequential queue
while tasklist //FI "PID eq $1" 2>/dev/null | grep -q python; do sleep 20; done
run() {  # ds lf extra_flag tag
  if done_m "$1" "$4"; then echo "[skip] $1 $4"; return; fi
  python -u models/embed_xgb.py --ds $1 --init $FM --label-frac $2 $3
}
for ds in paysim ieee_cis; do run $ds 1.0 --canonical-only "xgb+emb[fm_all]_canon"; done
for lf in 1.0 0.1 0.01; do
  s=""; [ $lf != 1.0 ] && s="_lf$lf"
  for ds in $SETS; do
    # trimmed 2026-11-17: low-label xgb+emb skipped on the three big sets (~1 h each); small sets cover that regime
    if [ $lf != 1.0 ]; then case $ds in ibm_card|aml_hi_medium|aml_li_medium) continue;; esac; fi
    run $ds $lf "" "xgb+emb[fm_all]$s"
  done
done
echo CPU_QUEUE_DONE
