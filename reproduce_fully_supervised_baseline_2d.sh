#!/bin/bash

# This script runs baselines with regime 100%

set -e

REPS=10      
START_REP=0
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2
OPTIMIZER=sgd
LR=0.5

DATASETS=(
    GlaS
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        python train_sup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime 100 --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice
        python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/fully_sup/unet/inv_temp-1/regime-100/run-$REP --device $GPU
    done
done
