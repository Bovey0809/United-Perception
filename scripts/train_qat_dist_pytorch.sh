#!/bin/bash

# cmd example: sh train_qat_dist_pytorch.sh 2

UP=/workspaces/United-Perception/
MQB=/workspace/MQBench/

cfg=$UP/configs/quant/det/retinanet/retinanet-r18-improve_quant_trt_qat.yaml
timestamp=$(date +%Y%m%d_%H%M%S)
logdir=$UP/logs
mkdir -p $logdir

export PYTHONPATH=$UP:$PYTHONPATH
export PYTHONPATH=$MQB:$PYTHONPATH

python -m up train \
              --ng=$1 \
              --launch=pytorch \
              --config=$cfg \
              --fork-method=spawn \
              --display=10 > $logdir/train_qat_dist_${timestamp}.log 2>&1 &

echo "Training log will be saved to $logdir/train_qat_dist_${timestamp}.log"