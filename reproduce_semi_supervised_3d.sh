#!/bin/bash

# This script runs baselines with several semi-supervised regimes

set -e

REPS=10
START_REP=0  
GPU=0

BATCH_SIZE=1
EVAL_BATCH_SIZE=1
OPTIMIZER=sgd
LR=0.1

NETWORKS=(
    unet3d
)

DATASETS=(
    Atrial
)

REGIMES=(
    1
    2
    5
    10
    20
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
    for REGIME in ${REGIMES[@]}; do
        for NETWORK in ${NETWORKS[@]}; do
            for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                python train_sup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)"
                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/unet3d/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 20)"
            done
        done
    done
done
