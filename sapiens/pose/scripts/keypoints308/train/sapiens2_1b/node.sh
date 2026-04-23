#!/bin/bash

cd "$(dirname "$(realpath "$0")")/../../../.." || exit

#-------------------------------------------------------------------------------
DEVICES=0,1,2,3,4,5,6,7
# DEVICES=0

#-------------------------------------------------------------------------------
TASK="keypoints308"
DATASET='shutterstock_goliath_3po'
MODEL="sapiens2_1b_${TASK}_${DATASET}-1024x768"

CONFIG_FILE="configs/${TASK}/$DATASET/${MODEL}.py"
TRAIN_BATCH_SIZE_PER_GPU=7
# TRAIN_BATCH_SIZE_PER_GPU=2

#-------------------------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

#-------------------------------------------------------------------------------
OUTPUT_DIR="Outputs/${TASK}/train/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

#-------------------------------------------------------------------------------
OPTIONS="train_dataloader.batch_size=$TRAIN_BATCH_SIZE_PER_GPU"
OPTIONS="${OPTIONS}${LOAD_FROM:+ load_from=$LOAD_FROM}"
CMD_RESUME="${RESUME_FROM:+--resume $RESUME_FROM}"

export TF_CPP_MIN_LOG_LEVEL=2
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

#-------------------------------------------------------------------------------
if [ "$mode" = "debug" ]; then
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    TRAIN_BATCH_SIZE_PER_GPU=1
    OPTIONS="train_dataloader.batch_size=${TRAIN_BATCH_SIZE_PER_GPU} train_dataloader.num_workers=0 train_dataloader.persistent_workers=False"
    OPTIONS="${OPTIONS}${LOAD_FROM:+ load_from=$LOAD_FROM}"

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options ${OPTIONS} \
        ${CMD_RESUME}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS=$(echo $DEVICES | tr -s ',' ' ' | wc -w)

    LOG_FILE="${OUTPUT_DIR}/log.txt"
    mkdir -p ${OUTPUT_DIR}
    touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} 'tools/dist_train.sh' ${CONFIG_FILE} \
        ${NUM_GPUS} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options ${OPTIONS} \
        ${CMD_RESUME} \
        | tee ${LOG_FILE}
fi
