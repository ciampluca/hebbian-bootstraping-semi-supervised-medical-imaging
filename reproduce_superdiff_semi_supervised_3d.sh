#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the SWTA-T Hebbian unsupervised pretraining

set -e

REPS=5
START_REP=0  
GPU=0

BATCH_SIZE=8
EVAL_BATCH_SIZE=8
OPTIMIZER=adam
LR=0.01
UNSUP_WEIGHT=1
EPOCHS=200
VALIDATE_ITER=1
TIMESTAMP_DIFFUSION=1000

REGIMES=(
    1
    2
    5
    10
    20
)

ROUNDS=(
    3
    5
    10
)

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



# Train & Evaluate
for DATASET in ${DATASETS[@]}; do
        for REGIME in ${REGIMES[@]}; do
            for NETWORK in ${NETWORKS[@]}; do
                for ROUND in ${ROUNDS[@]}; do
                    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
                        case $DATASET in
                            Atrial)
                                PRETRAINED_WEIGHTS_PATH="./runs/Atrial/superdiff_unsup/$NETWORK/inv_temp-1/regime-100/run-0/checkpoints/last.pth"
                                python train_semi_superdiff_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(96, 96, 80)" --load_weights $PRETRAINED_WEIGHTS_PATH --timestamp_diffusion $TIMESTAMP_DIFFUSION --diff_rounds $ROUND --unsup_weight $UNSUP_WEIGHT
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/superdiff_$NETWORK/diff_rounds-$ROUND/regime-$REGIME/run-$REP --patch_size "(96, 96, 80)" --patch_overlap "(48, 48, 40)" --device $GPU                         
                                ;;                     
                            LiTS)
                                PRETRAINED_WEIGHTS_PATH="./runs/LiTS/superdiff_unsup/$NETWORK/inv_temp-$1/regime-100/run-0/checkpoints/last.pth"
                                python train_semi_superdiff_3d.py --dataset_name $DATASET --network $NETWORK --path_dataset $DATA_ROOT/$DATASET --path_root_exp $EXP_ROOT --regime $REGIME --batch_size $BATCH_SIZE --optimizer $OPTIMIZER --seed $REP --validate_iter $VALIDATE_ITER --device $GPU --lr $LR --loss dice --patch_size "(112, 112, 32)" --load_weights $PRETRAINED_WEIGHTS_PATH --timestamp_diffusion $TIMESTAMP_DIFFUSION --diff_rounds $ROUND --unsup_weight $UNSUP_WEIGHT  --samples_per_volume_train 8 --samples_per_volume_val 12 
                                python test_3d.py --dataset_name $DATASET --network $NETWORK --batch_size $EVAL_BATCH_SIZE --path_dataset $DATA_ROOT/$DATASET --best JI --path_exp $EXP_ROOT/$DATASET/semi_sup/superdiff_$NETWORK/diff_rounds-$ROUND/regime-$REGIME/run-$REP --patch_size "(112, 112, 32)" --patch_overlap "(56, 56, 16)" --device $GPU --postprocessing True --fill_hole_thr 100
                                ;;
                        esac
                    done
                done
            done
        done
    done
done
