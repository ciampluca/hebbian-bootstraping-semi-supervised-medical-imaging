#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=5
START_REP=0     
GPU=0

DATASETS=(
    GlaS
)

K_VALUES=(
    1
    5
    10
    20
    50
    100
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



# Train & Evaluate
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
            for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                HEBBIAN_WEIGHTS_PATH="./runs/GlaS/hebbian_unsup/unet_swta_t/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                python train_sup.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size 2 --optimizer sgd --seed $REP --validate_iter 2 --device $GPU --lr 0.5 --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule swta_t --hebb_inv_temp $K  
            done
        done
    done
done


# Test 
# TODO