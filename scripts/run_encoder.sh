#!/bin/bash
# DeltaHippo on a bidirectional text encoder.
# usage: bash scripts/run_encoder.sh BENCH MODEL [SEED]
#   BENCH: tacred | fewrel | banking77 | clinc150
#   MODEL: bert-base-uncased | roberta-base
#          (read from ./hf_models/MODEL if that directory exists, otherwise from the Hugging Face Hub)
set -e
cd "$(dirname "$0")/.."
BENCH=$1; MODEL=$2; SEED=${3:-42}
CFG=configs/encoder/${BENCH}.yaml
[ -f "$CFG" ] || { echo "unknown benchmark $BENCH"; exit 1; }
export WANDB_MODE=disabled PDR_ALLOC=expandable_segments:True
NAME=deltahippo_enc_${BENCH}_$(echo $MODEL | tr -d '.-')_s${SEED}
mkdir -p experiments/logs
python main_CL.py --exp_prefix ${NAME} --cfg ${CFG} \
  --backbone ${MODEL} --classifier Linear --training_epochs 3 \
  --info_per_steps 25 --seed ${SEED} > experiments/logs/${NAME}.log 2>&1
grep "Test_Acc_Task_Seen" experiments/logs/${NAME}.log | tail -1
