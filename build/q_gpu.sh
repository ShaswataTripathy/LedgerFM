#!/bin/sh
# GPU queue: fine-tune / scratch / frozen-probe runs. Skips tags already in results/finetune.jsonl.
cd "$(dirname "$0")/.."
FM=checkpoints/fm_all/best.pt
SETS="sparkov paysim aml_hi_small aml_li_small ibm_card aml_hi_medium aml_li_medium ieee_cis"
done_tag() { grep -qF "\"tag\": \"$1\"" results/finetune.jsonl 2>/dev/null; }
sfx() { case $1 in 1.0) echo "";; *) echo "_lf$1";; esac; }
for lf in 1.0 0.1 0.01; do
  for ds in $SETS; do
    for init in fm_all scratch; do
      tag="${ds}__${init}$(sfx $lf)"
      if done_tag "$tag"; then echo "[skip] $tag"; continue; fi
      if [ $init = fm_all ]; then I=$FM; else I=scratch; fi
      python -u models/finetune.py --ds $ds --init $I --label-frac $lf
    done
  done
done
for ds in $SETS; do
  done_tag "${ds}__fm_all_frz" || python -u models/finetune.py --ds $ds --init $FM --freeze
done
echo GPU_QUEUE_DONE
