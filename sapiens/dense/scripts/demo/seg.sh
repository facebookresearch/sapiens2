#!/bin/bash
# Run body-part segmentation on a directory of images.

cd "$(dirname "$(realpath "$0")")/../.." || exit
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT:-${HOME}/sapiens2_host}"

#----------------------------set your input and output directories-------------------------
INPUT='./demo/data/itw_videos/reel1'
OUTPUT="${HOME}/Desktop/sapiens2/seg/Outputs/vis/itw_videos/reel1"

#--------------------------MODEL CARD (uncomment one)---------------------------------------
# MODEL_NAME='sapiens2_0.4b'; CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/sapiens2_0.4b_seg.safetensors"
# MODEL_NAME='sapiens2_0.8b'; CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/sapiens2_0.8b_seg.safetensors"
MODEL_NAME='sapiens2_1b';   CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/sapiens2_1b_seg.safetensors"
# MODEL_NAME='sapiens2_5b';   CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/seg/sapiens2_5b_seg.safetensors"

DATASET='shutterstock_goliath'
MODEL="${MODEL_NAME}_seg_${DATASET}-1024x768"
CONFIG_FILE="configs/seg/${DATASET}/${MODEL}.py"
OUTPUT="${OUTPUT}/${MODEL_NAME}"

CLASS_PALETTE="dome29"

##-------------------------------------inference--------------------------------------------
RUN_FILE='tools/vis/vis_seg.py'

JOBS_PER_GPU=3; GPU_IDS=(0 1 2 3 4 5 6 7)
# JOBS_PER_GPU=1; GPU_IDS=(0)
TOTAL_JOBS=$((JOBS_PER_GPU * ${#GPU_IDS[@]}))

IMAGE_LIST="${INPUT}/image_list.txt"
find "${INPUT}" -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) | sort > "${IMAGE_LIST}"

if [ ! -s "${IMAGE_LIST}" ]; then
  echo "No images found at ${INPUT}"
  exit 1
fi

NUM_IMAGES=$(wc -l < "${IMAGE_LIST}")
IMAGES_PER_FILE=$((NUM_IMAGES / TOTAL_JOBS))
EXTRA_IMAGES=$((NUM_IMAGES % TOTAL_JOBS))

export TF_CPP_MIN_LOG_LEVEL=2
echo "Distributing ${NUM_IMAGES} image paths into ${TOTAL_JOBS} jobs."

current_line=1
for ((i=0; i<TOTAL_JOBS; i++)); do
  TEXT_FILE="${INPUT}/image_paths_$((i+1)).txt"
  if [ $i -lt $EXTRA_IMAGES ]; then
    images_for_this_job=$((IMAGES_PER_FILE + 1))
  else
    images_for_this_job=$IMAGES_PER_FILE
  fi
  if [ $images_for_this_job -gt 0 ]; then
    sed -n "${current_line},$((current_line + images_for_this_job - 1))p" "${IMAGE_LIST}" > "${TEXT_FILE}"
    current_line=$((current_line + images_for_this_job))
  else
    touch "${TEXT_FILE}"
  fi
done

for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
  CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python ${RUN_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --save_pred \
    --input \"${INPUT}/image_paths_$((i+1)).txt\" \
    --output \"${OUTPUT}\""
  [ "$TOTAL_JOBS" -gt 1 ] && CMD="$CMD &"
  eval $CMD
  sleep 1
done

wait

rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/image_paths_$((i+1)).txt"
done

echo "Output directory: ${OUTPUT}"
