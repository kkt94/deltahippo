#!/bin/bash
# DeltaHippo on a decoder LLM.
# usage: bash scripts/run_llm.sh BENCH MODEL [SEED]
#   BENCH: tacred | fewrel | banking77 | clinc150
#   MODEL: Qwen3-0.6B | Qwen3-4B | Qwen3-8B | Llama-3.2-1B | Llama-3.2-3B | Llama-3.1-8B
#          (the checkpoint must be in ./MODEL, see README)
# The configuration follows the architecture rules of the paper:
#   * Qwen3-0.6B, Llama-3.2-1B, Llama-3.2-3B: configs/llm/BENCH.yaml
#   * Qwen3-4B (widest synapse-group input > 9,000): configs/llm/BENCH_wide.yaml (fp64 orthonormal basis)
#   * Qwen3-8B, Llama-3.1-8B: configs/llm/BENCH_8b.yaml (fp64 orthonormal basis + gradient checkpointing)
#   The width rule (hippo_widthref: 2560) is in every configuration; it only acts on models wider than 2560.
set -e
cd "$(dirname "$0")/.."
BENCH=$1; MODEL=$2; SEED=${3:-42}
case $MODEL in
  Qwen3-0.6B|Llama-3.2-1B|Llama-3.2-3B) CFG=configs/llm/${BENCH}.yaml ;;
  Qwen3-4B)                             CFG=configs/llm/${BENCH}_wide.yaml ;;
  Qwen3-8B|Llama-3.1-8B)                CFG=configs/llm/${BENCH}_8b.yaml ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "unknown benchmark $BENCH"; exit 1; }
export WANDB_MODE=disabled PDR_ALLOC=expandable_segments:True
NAME=deltahippo_${BENCH}_$(echo $MODEL | tr -d '.-')_s${SEED}
mkdir -p experiments/logs
python main_CL.py --exp_prefix ${NAME} --cfg ${CFG} \
  --backbone ${MODEL} --classifier None --training_epochs 3 \
  --info_per_steps 25 --seed ${SEED} > experiments/logs/${NAME}.log 2>&1
grep "Test_Acc_Task_Seen" experiments/logs/${NAME}.log | tail -1
