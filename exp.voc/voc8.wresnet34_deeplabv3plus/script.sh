#!/usr/bin/env bash

export volna="/path/to/DATA"
export NGPUS=4
export OUTPUT_PATH="/path/to/result_dir"
export snapshot_dir=$OUTPUT_PATH/snapshot

export batch_size=8
export learning_rate=0.001
export snapshot_iter=1

python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py 
export TARGET_DEVICE=3
python3 eval.py -e 20-34 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results
