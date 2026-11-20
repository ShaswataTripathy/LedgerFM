#!/bin/sh
# Fresh test world aml_hi_large: val-only selection, then ONE test run. Usage: sh build/q_fresh.sh cpu|gpu
cd "$(dirname "$0")/.."
D=aml_hi_large
done_ft() { grep -qF "\"tag\": \"$1\"" results/finetune.jsonl 2>/dev/null; }
done_x() { grep -F "\"ds\": \"$D\", \"model\": \"$1\"," results/baselines.jsonl >/dev/null 2>&1; }
case $1 in
cpu)
  done_x xgb || python -u models/baseline_xgb.py $D
  done_x xgb_lf0.1 || python -u models/baseline_xgb.py $D --label-frac 0.1
  echo FRESH_CPU_DONE;;
gpu)
  while [ ! -f checkpoints/fm_v01/last.pt ] || ! grep -q '"step": 25000' checkpoints/fm_v01/log.jsonl 2>/dev/null; do sleep 60; done
  for job in "fm_v01/last.pt 1.0" "fm_v01/last.pt 0.1" "fm_all/best.pt 1.0" "scratch 1.0"; do
    set -- $job
    if [ $1 = scratch ]; then I=scratch; n=scratch; else I=checkpoints/$1; n=${1%%/*}; fi
    s=""; [ $2 != 1.0 ] && s="_lf$2"
    done_ft "${D}__$n$s" || python -u models/finetune.py --ds $D --init $I --label-frac $2
  done
  echo FRESH_GPU_DONE;;
esac
