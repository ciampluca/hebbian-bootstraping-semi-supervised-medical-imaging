#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
START_REP=0
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2
LR=0.5
OPTIMIZER=adam
THRESHOLD=0.5

K_VALUES=(
    1
    5
    10
    20
    50
    100
)

DATASETS=(
    GlaS
)

HEBB_MODES=(
    swta_t
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for HEBB_MODE in ${HEBB_MODES[@]}; do
            python pretrain_hebbian_unsup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter 2 --device $GPU --lr $LR --loss dice --hebb_mode $HEBB_MODE --hebb_inv_temp $K --threshold $THRESHOLD
            python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/hebbian_unsup/unet_$HEBB_MODE/inv_temp-$K/regime-100/run-0 --hebbian_pretrain True
        done
    done
done