#!/bin/sh
# CPU queue, small sets only: safe to run beside the GPU queue.
# aml_hi_medium / aml_li_medium xgb+emb stay in q_cpu.sh; run that once the GPU queue is finished (31.9M rows, OOM risk).
cd "$(dirname "$0")/.."
FM=checkpoints/fm_all/best.pt
SETS="sparkov paysim aml_hi_small aml_li_small ieee_cis"
done_m() { grep -F "\"ds\": \"$1\", \"model\": \"$2\"," results/baselines.jsonl >/dev/null 2>&1; }
run() {  # ds lf tag
  if done_m "$1" "$3"; then echo "[skip] $1 $3"; return; fi
  python -u models/embed_xgb.py --ds $1 --init $FM --label-frac $2
}
for lf in 1.0 0.1 0.01; do
  s=""; [ $lf != 1.0 ] && s="_lf$lf"
  for ds in $SETS; do run $ds $lf "xgb+emb[fm_all]$s"; done
done
echo CPU_SMALL_QUEUE_DONE
