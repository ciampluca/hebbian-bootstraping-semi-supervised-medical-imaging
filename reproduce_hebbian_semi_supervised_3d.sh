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
UNSUP_WEIGHT=5
VALIDATE_ITER=1

K_VALUES=(
    1
    5
    10
    20
    50
    75
    100
)

REGIMES=(
    1
    2
    5
    10
    20
)

NETWORKS=(
    unet3d
    # vnet
)

DATASETS=(
    Atrial
    LiTS
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
            for NETWORK in ${NETWORKS[@]}; do
                for HEBB_MODE in ${HEBB_MODES[@]}; do
                    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                        case $DATASET in
                            Atrial)
                                HEBBIAN_WEIGHTS_PATH="./runs/Atrial/hebbian_unsup/$NETWORK"_swta_t"/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                                HEBBIAN_URPC_WEIGHTS_PATH="./runs/Atrial/hebbian_unsup/$NETWORK"_urpc_swta_t"/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                                HEBBIAN_DTC_WEIGHTS_PATH="./runs/Atrial/hebbian_unsup/$NETWORK"_dtc_swta_t"/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                                HEBBIAN_CCT_WEIGHTS_PATH="./runs/Atrial/hebbian_unsup/$NETWORK"_cct_swta_t"/inv_temp-$K/regime-100/run-0/checkpoints/last.pth"
                                python train_sup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU
                                python train_semi_EM_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_em_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU                    
                                python train_semi_UAMT_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_uamt_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU                        
                                python train_semi_CPS_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_cps_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU                         
                                python train_semi_URPC_3d.py --dataset_name $DATASET --network $NETWORK"_urpc" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_URPC_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK"_urpc" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_urpc_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU                         
                                python train_semi_CCT_3d.py --dataset_name $DATASET --network $NETWORK"_cct" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_CCT_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK"_cct" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_cct_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU                                   
                                python train_semi_DTC_3d.py --dataset_name $DATASET --network $NETWORK"_dtc" --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_hebbian_weights $HEBBIAN_DTC_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK"_dtc" --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_dtc_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --hebbian_pretrain True --device $GPU
                                ;;                     
                            LiTS)
                                python train_sup_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(112, 112, 32)" --load_hebbian_weights $HEBBIAN_WEIGHTS_PATH --hebbian_rule $HEBB_MODE --hebb_inv_temp $K  --samples_per_volume_train 8 --samples_per_volume_val 12 --num_epochs 250
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/h_"$NETWORK"_"$HEBB_MODE"/inv_temp-$K/regime-$REGIME/run-$REP --patch_size "(112, 112, 32)" --patch_overlap "(56, 56, 16)" --hebbian_pretrain True --device $GPU --postprocessing True --fill_hole_thr 100
                                ;;
                        esac
                    done
                done
            done
        done
    done
done