#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
GPU=0

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

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate 
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
            python pretrain_hebbian_unsup.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size 2 --optimizer adam --seed $REP --validate_iter 2 --device $GPU --lr 0.5 --loss dice --hebb_mode swta_t --hebb_inv_temp $K
        done
    done
done


# Test 
# TODO