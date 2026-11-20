#!/bin/sh
# IEEE-CIS (eval-only real data). XGB part runs now (CPU); GPU part waits for phase 3 to finish.
cd "$(dirname "$0")/.."
python -u models/baseline_xgb.py ieee_cis
python -u models/baseline_xgb.py ieee_cis --canonical-only
python -u models/baseline_xgb.py ieee_cis --label-frac 0.1
python -u models/baseline_xgb.py ieee_cis --label-frac 0.01
until grep -q PHASE3_DONE logs/phase3.log; do sleep 60; done
FM=checkpoints/fm_all/best.pt
for lf in 1.0 0.1 0.01; do
  python -u models/finetune.py --ds ieee_cis --init $FM --label-frac $lf
  python -u models/finetune.py --ds ieee_cis --init scratch --label-frac $lf
  python -u models/embed_xgb.py --ds ieee_cis --init $FM --label-frac $lf
done
python -u models/finetune.py --ds ieee_cis --init $FM --freeze
echo IEEE_DONE
# PaySim's raw balance columns leak the label, so its fair embedding test is canonical-only + embeddings.
python -u models/embed_xgb.py --ds paysim --canonical-only
python -u models/embed_xgb.py --ds ieee_cis --canonical-only
echo EXTRA_DONE
