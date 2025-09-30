#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=10
START_REP=0     
GPU=0

BATCH_SIZE=8
EVAL_BATCH_SIZE=8
OPTIMIZER=adam
LR=0.01
UNSUP_WEIGHT=5
VALIDATE_ITER=1

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
    OCT-CME
    QaTa-COV19
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
                for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                    python train_snn_sup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice  
                    python test_snn_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/snn/inv_temp-1/regime-$REGIME/run-$REP --device $GPU
                done
        done
done

