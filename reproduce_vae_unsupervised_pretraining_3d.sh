#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
START_REP=0
GPU=0

BATCH_SIZE=1
EVAL_BATCH_SIZE=1
LR=0.0001
OPTIMIZER=adam
THRESHOLD=0.5
VALIDATE_ITER=2

NETWORKS=(
    unet3d_vae
)

DATASETS=(
    Atrial
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
    for NETWORK in ${NETWORKS[@]}; do
            case $DATASET in
                Atrial)
                    python pretrain_vae_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" #--threshold $THRESHOLD
                    python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best last --path_exp $EXP_ROOT/$DATASET/vae_unsup/$NETWORK/inv_temp-1/regime-100/run-0 --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU #--threshold $THRESHOLD
                    ;;
            esac
        done
done
