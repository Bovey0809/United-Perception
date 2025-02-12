#!/bin/bash

# cmd example: sh train_qat.sh 1 8 ToolChain

UP=/workspaces/United-Perception/
MQB=/workspace/MQBench/

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml
timestamp=$(date +%Y%m%d_%H%M%S)
logdir=$UP/logs
mkdir -p $logdir

jobname=quant_qat

export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH

srun -N$1 --gres=gpu:$2 -p $3 --job-name=$jobname --cpus-per-task=2 \
nohup python -u -m up train \
  --ng=$2 \
  --launch=pytorch \
  --config=$cfg \
  --display=10 \
  > $logdir/train_qat_${timestamp}.log 2>&1 &

echo "Training log will be saved to $logdir/train_qat_${timestamp}.log"