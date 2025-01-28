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

NETWORKS=(
    unet_ddpm
    # vnet
)

DATASETS=(
    Atrial
    LiTS
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
        for NETWORK in ${NETWORKS[@]}; do
                case $DATASET in
                    Atrial)
                        python pretrain_superdiff_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --timestamp_diffusion $TIMESTAMP_DIFFUSION --threshold $THRESHOLD
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best last --path_exp $EXP_ROOT/$DATASET/superdiff_unsup/$NETWORK/inv_temp-1/regime-100/run-0 --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --timestamp_diffusion $TIMESTAMP_DIFFUSION --device $GPU #--threshold $THRESHOLD
                        ;;  
                    LiTS)
                        python pretrain_superdiff_unsup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed 0 --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(112, 112, 32)" --samples_per_volume_train 8 --samples_per_volume_val 12 --timestamp_diffusion $TIMESTAMP_DIFFUSION --threshold $THRESHOLD
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best last --path_exp $EXP_ROOT/$DATASET/superdiff_unsup/$NETWORK/inv_temp-1/regime-100/run-0 --patch_size "(112, 112, 32)" --patch_overlap "(56, 56, 16)" --device $GPU --postprocessing True --fill_hole_thr 100 --timestamp_diffusion $TIMESTAMP_DIFFUSION #--threshold $THRESHOLD
                        ;;
                esac
        done
done
