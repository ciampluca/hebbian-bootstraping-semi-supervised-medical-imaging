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
UNSUP_WEIGHT=1
EPOCHS=200
VALIDATE_ITER=1
TIMESTAMP_DIFFUSION=1000


REGIMES=(
    1
    2
    5
    10
    20
    #100
)

ROUNDS=(
    3
    5
    10
)

DATASETS=(
    GlaS
    PH2
    HMEPS
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
            for ROUND in ${ROUNDS[@]}; do
                for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                    PRETRAINED_WEIGHTS_PATH="./runs/$DATASET/superdiff_unsup/unet_ddpm/inv_temp-1/regime-100/run-0/checkpoints/last.pth"
                    python train_semi_superdiff_2d.py --dataset_name $DATASET --network unet_ddpm --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME -e $EPOCHS --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --load_weights $PRETRAINED_WEIGHTS_PATH --timestamp_diffusion $TIMESTAMP_DIFFUSION --diff_rounds $ROUND --unsup_weight $UNSUP_WEIGHT
                    python test_2d.py --dataset_name $DATASET --network unet_ddpm --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/superdiff_unet_ddpm/diff_rounds-$ROUND/regime-$REGIME/run-$REP --device $GPU --timestamp_diffusion $TIMESTAMP_DIFFUSION
                done
            done
        done
done
