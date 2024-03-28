#!/bin/bash

# This script runs baselines with regime 100%

set -e

REPS=1      # set a 1 for now, how can we have multiple runs in this case?
START_REP=0
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2
BATCH_SIZE_3D=1
EVAL_BATCH_SIZE_3D=1

DATASETS_2D=(
    GlaS
)

DATASETS_3D=(
    Atrial
)

REGIMES=(
    100
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS_2D[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        python train_sup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime 100 --batch_size $BATCH_SIZE --optimizer sgd --seed $REP --validate_iter 2 --device $GPU --lr 0.5 --loss dice
        python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/fully_sup/unet/inv_temp-1/regime-100/run-$REP
    done
done

for DATASET in ${DATASETS_3D[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        python train_sup_3d.py --dataset_name $DATASET --network unet3d --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime 100 --batch_size $BATCH_SIZE_3D --optimizer sgd --seed $REP --validate_iter 2 --device $GPU --lr 0.1 --loss dice --patch_size "(96, 96, 80)"
        python test_3d.py --dataset_name $DATASET --network unet3d --batch_size $EVAL_BATCH_SIZE_3D --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/fully_sup/unet3d/inv_temp-1/regime-100/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 20)"
    done
done