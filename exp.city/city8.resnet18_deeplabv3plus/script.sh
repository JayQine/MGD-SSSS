#!/usr/bin/env bash
#nvidia-smi

export volna="../../"
export NGPUS=8
export OUTPUT_PATH="./results"
export snapshot_dir=$OUTPUT_PATH/snapshot

export NGPUS=8
export learning_rate=0.02
export batch_size=16
export snapshot_iter=5

python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py 
export TARGET_DEVICE=7
python3 eval.py -e 80-137 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results


# following is the command for debug
#export NGPUS=1
#export batch_size=2
#python3 -m torch.launch --nproc_per_node=$NGPUS --master_port 29005 train.py --debug 1