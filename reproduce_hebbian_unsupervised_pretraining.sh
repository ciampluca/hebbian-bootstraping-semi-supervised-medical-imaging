#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
START_REP=0
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2

K_VALUES=(
    1
    5
    10
    20
    50
    100
)

DATASETS_2D=(
    GlaS
)

DATASETS_3D=(
    Atrial
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS_2D[@]}; do
        python pretrain_hebbian_unsup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer adam --seed 0 --validate_iter 2 --device $GPU --lr 0.5 --loss dice --hebb_mode swta_t --hebb_inv_temp $K --threshold 0.5
        python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/hebbian_unsup/unet/inv_temp-$K/regime-100/run-0
    done
done