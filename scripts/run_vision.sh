#!/bin/bash
# DeltaHippo on vision models (class-incremental, 10 tasks, no task id).
# usage: bash scripts/run_vision.sh MODEL DATASET [SEED]
#   MODEL:   vit | resnet50
#   DATASET: cifar100 | imr        (ImageNet-R; build its cache first with tools/vision_data_prep.py)
#   SEED:    the reported runs used 1993 (the default), 42 and 3
set -e
cd "$(dirname "$0")/.."
MODEL=$1; DATA=$2; SEED=${3:-1993}
COMMON="--method ours --tasks 10 --epochs 3 --bs 64 --fresh_head --oldrow --oldrow_span --fresh_aug --wdfold --pdet --cmpowm"
case $MODEL in
  vit)      FLAGS="$COMMON --lastcls --tabln --sink --lr_bb 3e-5" ;;
  resnet50) FLAGS="$COMMON --no_owm --cmpcons" ;;          # backbone lr: the default 1e-4
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
NAME=deltahippo_${MODEL}_${DATA}_s${SEED}
mkdir -p experiments/logs experiments/vision
python tools/vision_hippo.py --model ${MODEL} --dataset ${DATA} --seed ${SEED} ${FLAGS} \
  --out experiments/vision/${NAME} > experiments/logs/${NAME}.log 2>&1
grep "FINAL AA" experiments/logs/${NAME}.log | tail -1
