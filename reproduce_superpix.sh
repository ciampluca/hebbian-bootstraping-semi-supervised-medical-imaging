#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=1      # 1 because only pretraining
START_REP=0
GPU=0

BATCH_SIZE=1
EVAL_BATCH_SIZE=1
LR=0.00001
OPTIMIZER=adam
THRESHOLD=0.5
VALIDATE_ITER=2

NETWORKS=(
    unet3d
    # vnet
)

K_VALUES=(
    1
)

DATASETS=(
    Atrial
    #LiTS
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for NETWORK in ${NETWORKS[@]}; do
                case $DATASET in
                    Atrial)
                        python pretrain_hebbian_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" #--threshold $THRESHOLD
                        ;;  
                    LiTS)
                        python pretrain_hebbian_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(112, 112, 32)" --samples_per_volume_train 8 --samples_per_volume_val 12 #--threshold $THRESHOLD
                        ;;
                esac
            done
    done
done
