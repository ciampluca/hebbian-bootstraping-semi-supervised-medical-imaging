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

K_VALUES=(
    1
    5
    10
    20
    50
    100
)

REGIMES=(
    1
    2
    5
    10
    20
)

DATASETS=(
    GlaS
)

HEBB_MODES=(
    swta_t
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Evaluate
for K in ${K_VALUES[@]}; do
    for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
            for HEBB_MODE in ${HEBB_MODES[@]}; do
                for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                    HEBBIAN_WEIGHTS_PATH="./runs/GlaS/hebbian_unsup/unet_swta_t/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                    python train_sup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  
                    python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_unet_$HEBB_MODE/inv_temp-$K/regime-$REGIME/run-$REP --hebbian_pretrain True --device $GPU
                    python train_semi_EM_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                    python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_em_unet_$HEBB_MODE/inv_temp-$K/regime-$REGIME/run-$REP --hebbian_pretrain True --device $GPU
                    python train_semi_UAMT_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                    python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_uamt_unet_$HEBB_MODE/inv_temp-$K/regime-$REGIME/run-$REP --hebbian_pretrain True --device $GPU                   
                    python train_semi_CPS_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                    python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_cps_unet_$HEBB_MODE/inv_temp-$K/regime-$REGIME/run-$REP --hebbian_pretrain True --device $GPU  
                    python train_semi_URPC_2d.py --dataset_name $DATASET --network unet_urpc --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                    python test_2d.py --dataset_name $DATASET --network unet_urpc --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_urpc_unet_$HEBB_MODE/inv_temp-$K/regime-$REGIME/run-$REP --hebbian_pretrain True --device $GPU  
                done
            done
        done
    done
done