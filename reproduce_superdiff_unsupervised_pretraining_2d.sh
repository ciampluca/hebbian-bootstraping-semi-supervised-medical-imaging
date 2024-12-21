#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
START_REP=0
GPU=0

BATCH_SIZE=8
EVAL_BATCH_SIZE=8
LR=0.001
OPTIMIZER=adam
THRESHOLD=0.5
VALIDATE_ITER=1
TIMESTAMP_DIFFUSION=1000

DATASETS=(
    GlaS
    PH2
    HMEPS
)


DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
        python pretrain_superdiff_unsup_2d.py --dataset_name $DATASET --network unet_ddpm --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --timestamp_diffusion $TIMESTAMP_DIFFUSION --threshold $THRESHOLD
        python test_2d.py --dataset_name $DATASET --network unet_ddpm --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/ddpm_unsup/unet_ddpm/inv_temp-1/regime-100/run-0 --device $GPU --timestamp_diffusion $TIMESTAMP_DIFFUSION #--threshold $THRESHOLD 
done
