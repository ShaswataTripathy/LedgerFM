#!/bin/sh
# 3-seed check of the close calls (seed 0 = existing runs; this adds seeds 1, 2). Skips finished tags.
# Same seed => same low-label subset for FM and XGBoost (paired comparison).
cd "$(dirname "$0")/.."
FM=checkpoints/fm_all/best.pt
done_ft() { grep -qF "\"tag\": \"$1\"" results/finetune.jsonl 2>/dev/null; }
done_x() { grep -F "\"ds\": \"$1\", \"model\": \"$2\"," results/baselines.jsonl >/dev/null 2>&1; }
sfx() { [ $1 = 1.0 ] && echo "" || echo "_lf$1"; }
gpu() {
  for s in 1 2; do
    for job in "aml_li_small fm_all 1.0" "aml_li_small fm_all 0.1" "aml_li_small fm_all 0.01" \
               "aml_hi_small fm_all 1.0" "aml_hi_small fm_all 0.1" \
               "aml_li_small loo_aml_li 1.0" "aml_li_small loo_aml_li 0.1"; do
      set -- $job; t="$1__$2$(sfx $3)_s$s"
      done_ft "$t" && { echo "[skip] $t"; continue; }
      python -u models/finetune.py --ds $1 --init checkpoints/$2/best.pt --label-frac $3 --seed $s
    done
  done; echo GPU_SEEDS_DONE
}
cpu() {
  for s in 1 2; do
    for job in "aml_hi_small 1.0" "aml_hi_small 0.1" "aml_li_small 1.0" "aml_li_small 0.1" "aml_li_small 0.01"; do
      set -- $job
      done_x $1 "xgb$(sfx $2)_s$s" || python -u models/baseline_xgb.py $1 --label-frac $2 --seed $s
      done_x $1 "xgb+emb[fm_all]$(sfx $2)_s$s" || python -u models/embed_xgb.py --ds $1 --init $FM --label-frac $2 --seed $s
    done
  done; echo CPU_SEEDS_DONE
}
case $1 in gpu) gpu;; cpu) cpu;; esac
