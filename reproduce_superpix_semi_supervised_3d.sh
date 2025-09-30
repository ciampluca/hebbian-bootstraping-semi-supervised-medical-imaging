#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=5
START_REP=0  
GPU=0

BATCH_SIZE=1
EVAL_BATCH_SIZE=1
OPTIMIZER=sgd
LR=0.1
VALIDATE_ITER=1

REGIMES=(
    1
    2
    5
    10
    20
)

NETWORKS=(
    unet3d_superpix
    # vnet
)

DATASETS=(
    Atrial
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
            for NETWORK in ${NETWORKS[@]}; do
                    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                        case $DATASET in
                            Atrial)
                                WEIGHTS_PATH="./runs/Atrial/superpix_unsup/$NETWORK/inv_temp-1/regime-100/run-0/checkpoints/last.pth"
                                python train_sup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_weights $WEIGHTS_PATH  
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU
                                ;;                     
                        esac
                    done
            done
        done
done
