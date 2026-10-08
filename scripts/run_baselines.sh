#!/bin/bash
# Re-implemented baselines under the same full fine-tuning protocol as DeltaHippo (decoder LLMs).
# usage: bash scripts/run_baselines.sh METHOD BENCH MODEL [SEED]
#   METHOD: sculpt (Sculpting Subspaces, fixed rank budget) | codecl (CODE-CL, task-agnostic K = 0)
#   BENCH:  tacred | fewrel | banking77 | clinc150
#   MODEL:  as in scripts/run_llm.sh
set -e
cd "$(dirname "$0")/.."
METHOD=$1; BENCH=$2; MODEL=$3; SEED=${4:-42}
CFG=configs/baselines/${METHOD}_${BENCH}.yaml
[ -f "$CFG" ] || { echo "unknown method/benchmark ${METHOD}/${BENCH}"; exit 1; }
export WANDB_MODE=disabled PDR_ALLOC=expandable_segments:True
NAME=${METHOD}_${BENCH}_$(echo $MODEL | tr -d '.-')_s${SEED}
mkdir -p experiments/logs
python main_CL.py --exp_prefix ${NAME} --cfg ${CFG} \
  --backbone ${MODEL} --classifier None --training_epochs 3 \
  --info_per_steps 25 --seed ${SEED} > experiments/logs/${NAME}.log 2>&1
grep "Test_Acc_Task_Seen" experiments/logs/${NAME}.log | tail -1
