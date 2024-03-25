#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # set a 1 for now, how can we have multiple runs in this case?
GPU=0

DATASETS=(
    GlaS
)

REGIMES=(
    100
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        python train_sup.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime 100 --batch_size 2 --optimizer sgd --seed $REP --validate_iter 2 --device $GPU --lr 0.5 --loss dice
    done
done


# Test 
# TODO