#!/bin/sh
# Leave-one-dataset-out: pretrain without the held-out set(s), then fine-tune on them. Skips finished work.
cd "$(dirname "$0")/.."
done_tag() { grep -qF "\"tag\": \"$1\"" results/finetune.jsonl 2>/dev/null; }
pre() {  # run sets
  if [ -f checkpoints/$1/best.pt ] && grep -q '"step": 25000' checkpoints/$1/log.jsonl 2>/dev/null; then echo "[skip] pretrain $1"; return; fi
  python -u models/pretrain.py --run $1 --sets $2 --bs 128 --steps 25000
}
ft() {  # run ds
  for lf in 1.0 0.1 0.01; do
    s=""; [ $lf != 1.0 ] && s="_lf$lf"
    done_tag "${2}__$1$s" && { echo "[skip] ${2}__$1$s"; continue; }
    python -u models/finetune.py --ds $2 --init checkpoints/$1/best.pt --label-frac $lf
  done
}
pre loo_sparkov aml_hi_small,aml_li_small,aml_hi_medium,aml_li_medium,ibm_card
ft loo_sparkov sparkov
# both AML LI worlds held out together: keeping li_medium would leak the same generator settings into the li_small transfer test
pre loo_aml_li aml_hi_small,aml_hi_medium,ibm_card,sparkov
ft loo_aml_li aml_li_small
echo LOO_QUEUE_DONE
