#!/bin/bash

# This script runs baselines with several semi-supervised regimes

set -e

REPS=5
START_REP=0
GPU=0

BATCH_SIZE=1
EVAL_BATCH_SIZE=1
OPTIMIZER=sgd
LR=0.1
UNSUP_WEIGHT=5
VALIDATE_ITER=1

NETWORKS=(
    unet3d
    #vnet
)

INIT_WEIGHTS=(
    'kaiming'
    #'xavier'
    #'orthogonal'
)

DATASETS=(
    Atrial
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
        for NETWORK in ${NETWORKS[@]}; do
            for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                case $DATASET in
                    Atrial)
                        for INIT_WEIGHT in  ${INIT_WEIGHTS[@]}; do
                            python train_sup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter 2 --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --init_weights $INIT_WEIGHT
                            python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/$INIT_WEIGHT"_"$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU
                        done
                        python train_semi_EM_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT 
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/em_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU            
                        python train_semi_UAMT_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/uamt_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU               
                        python train_semi_CPS_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT
                        python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/cps_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU               
                        python train_semi_URPC_3d.py --dataset_name $DATASET --network $NETWORK"_urpc" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT
                        python test_3d.py --dataset_name $DATASET --network $NETWORK"_urpc" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/urpc_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU                
                        python train_semi_CCT_3d.py --dataset_name $DATASET --network $NETWORK"_cct" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT
                        python test_3d.py --dataset_name $DATASET --network $NETWORK"_cct" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/cct_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU             
                        python train_semi_DTC_3d.py --dataset_name $DATASET --network $NETWORK"_dtc" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --unsup_weight $UNSUP_WEIGHT
                        python test_3d.py --dataset_name $DATASET --network $NETWORK"_dtc" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/dtc_$NETWORK/inv_temp-1/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU             
                        ;;
                esac
            done
        done
    done
done
