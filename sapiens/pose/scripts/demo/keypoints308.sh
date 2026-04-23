#!/bin/bash
# Run 308-keypoint pose estimation on a directory of images.

cd "$(dirname "$(realpath "$0")")/../.." || exit
SAPIENS_CHECKPOINT_ROOT="${SAPIENS_CHECKPOINT_ROOT:-${HOME}/sapiens2_host}"

#----------------------------set your input and output directories-------------------------
INPUT='./demo/data/itw_videos/reel1'
OUTPUT="${HOME}/Desktop/sapiens2/pose/Outputs/vis/itw_videos/reel1"

#--------------------------MODEL CARD (uncomment one)---------------------------------------
# MODEL_NAME='sapiens2_0.4b'; CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/sapiens2_0.4b_pose.safetensors"
# MODEL_NAME='sapiens2_0.8b'; CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/sapiens2_0.8b_pose.safetensors"
MODEL_NAME='sapiens2_1b';   CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/sapiens2_1b_pose.safetensors"
# MODEL_NAME='sapiens2_5b';   CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/sapiens2_5b_pose.safetensors"

DATASET='shutterstock_goliath_3po'
MODEL="${MODEL_NAME}_keypoints308_${DATASET}-1024x768"
CONFIG_FILE="configs/keypoints308/${DATASET}/${MODEL}.py"
OUTPUT="${OUTPUT}/${MODEL_NAME}"

# Person detector (for bbox)
DETECTION_CONFIG_FILE='tools/vis/rtmdet_m_640-8xb32_coco-person.py'
DETECTION_CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/detector/rtmdet_m.pth"

#---------------------------VISUALIZATION PARAMS--------------------------------------------
LINE_THICKNESS=8
RADIUS=8
KPT_THRES=0.3

##-------------------------------------inference--------------------------------------------
RUN_FILE='tools/vis/vis_pose.py'

# Number of inference jobs per GPU and which GPUs to use
JOBS_PER_GPU=2; GPU_IDS=(0 1 2 3 4 5 6 7)
# JOBS_PER_GPU=1; GPU_IDS=(0)
TOTAL_JOBS=$((JOBS_PER_GPU * ${#GPU_IDS[@]}))

# Find images and partition across jobs
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

# Launch parallel inference
for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU_ID=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
  CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python ${RUN_FILE} \
    ${DETECTION_CONFIG_FILE} \
    ${DETECTION_CHECKPOINT} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --input \"${INPUT}/image_paths_$((i+1)).txt\" \
    --output \"${OUTPUT}\" \
    --radius ${RADIUS} \
    --kpt-thr ${KPT_THRES} \
    --thickness ${LINE_THICKNESS}"
  [ "$TOTAL_JOBS" -gt 1 ] && CMD="$CMD &"
  eval $CMD
  sleep 1
done

wait

# Cleanup
rm "${IMAGE_LIST}"
for ((i=0; i<TOTAL_JOBS; i++)); do
  rm "${INPUT}/image_paths_$((i+1)).txt"
done

echo "Output directory: ${OUTPUT}"
