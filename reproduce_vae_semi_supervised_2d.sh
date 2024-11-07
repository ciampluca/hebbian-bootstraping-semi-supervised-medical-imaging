#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=10
START_REP=0
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2
OPTIMIZER=sgd
LR=0.5
UNSUP_WEIGHT=5
VALIDATE_ITER=2

REGIMES=(
    1
    2
    5
    10
    20
)

DATASETS=(
    GlaS
    PH2
    HMEPS
)


DATA_ROOT=./data
EXP_ROOT=./



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
    for REGIME in ${REGIMES[@]}; do
            for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                WEIGHTS_PATH="./runs/$DATASET/vae_unsup/unet_vae/inv_temp-1/regime-100/run-0/checkpoints/last.pth"
                python train_sup_2d.py --dataset_name $DATASET --network unet_vae --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice
                python test_2d.py --dataset_name $DATASET --network unet_vae --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/unet_vae/inv_temp-1/regime-$REGIME/run-$REP --device $GPU
            done
    done
done
