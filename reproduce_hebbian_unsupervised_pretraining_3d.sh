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
    5
    10
    20
    50
    75
    100
)

DATASETS=(
    Atrial
    LiTS
)

HEBB_MODES=(
    swta_t
)

EXCLUDE_LAYER="conv"

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for NETWORK in ${NETWORKS[@]}; do
            for HEBB_MODE in ${HEBB_MODES[@]}; do
                case $DATASET in
                    Atrial)
                        python pretrain_hebbian_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --hebb_mode $HEBB_MODE --hebb_inv_temp $K --exclude $EXCLUDE_LAYER #--threshold $THRESHOLD
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best last --path_exp $EXP_ROOT/$DATASET/hebbian_unsup/$NETWORK"_"$HEBB_MODE/inv_temp-$K/regime-100/run-0 --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU #--threshold $THRESHOLD
                        python pretrain_hebbian_unsup_3d.py --dataset_name $DATASET --network $NETWORK"_"urpc --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --hebb_mode $HEBB_MODE --hebb_inv_temp $K --exclude $EXCLUDE_LAYER #--threshold $THRESHOLD
                        python test_3d.py --dataset_name $DATASET --network $NETWORK"_"urpc --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best last --path_exp $EXP_ROOT/$DATASET/hebbian_unsup/$NETWORK"_urpc_"$HEBB_MODE/inv_temp-$K/regime-100/run-0 --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU #--threshold $THRESHOLD          
                        ;;  
                    LiTS)
                        ;;
                esac
            done
        done
    done
done