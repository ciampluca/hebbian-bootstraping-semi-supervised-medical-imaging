#!/bin/bash

# This script runs baselines with several semi-supervised regimes

set -e

REPS=10
START_REP=0  
GPU=0

BATCH_SIZE=2
EVAL_BATCH_SIZE=2
OPTIMIZER=sgd
LR=0.5
UNSUP_WEIGHT=5
VALIDATE_ITER=1

INIT_WEIGHTS=(
    'kaiming'
    'xavier'
    'orthogonal'
)

DATASETS=(
    GlaS
    PH2
    HMEPS
)

REGIMES=(
    1
    2
    5
    10
    20
)

DATA_ROOT=./data
EXP_ROOT=./runs



# Train & Test
for DATASET in ${DATASETS[@]}; do
    for REGIME in ${REGIMES[@]}; do
        for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
            for INIT_WEIGHT in  ${INIT_WEIGHTS[@]}; do
                python train_sup_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --init_weights $INIT_WEIGHT
                python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/$INIT_WEIGHT"_unet"/inv_temp-1/regime-$REGIME/run-$REP --device $GPU
            done
            python train_semi_EM_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --unsup_weight $UNSUP_WEIGHT
            python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/em_unet/inv_temp-1/regime-$REGIME/run-$REP --device $GPU        
            python train_semi_UAMT_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --unsup_weight $UNSUP_WEIGHT
            python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/uamt_unet/inv_temp-1/regime-$REGIME/run-$REP --device $GPU          
            python train_semi_CPS_2d.py --dataset_name $DATASET --network unet --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --unsup_weight $UNSUP_WEIGHT
            python test_2d.py --dataset_name $DATASET --network unet --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/cps_unet/inv_temp-1/regime-$REGIME/run-$REP --device $GPU 
            python train_semi_URPC_2d.py --dataset_name $DATASET --network unet_urpc --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --unsup_weight $UNSUP_WEIGHT
            python test_2d.py --dataset_name $DATASET --network unet_urpc --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/urpc_unet/inv_temp-1/regime-$REGIME/run-$REP --device $GPU          
            python train_semi_CCT_2d.py --dataset_name $DATASET --network unet_cct --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --unsup_weight $UNSUP_WEIGHT
            python test_2d.py --dataset_name $DATASET --network unet_cct --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/cct_unet/inv_temp-1/regime-$REGIME/run-$REP --device $GPU         
        done
    done
done


