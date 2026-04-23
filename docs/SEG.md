# Sapiens2: Body-Part Segmentation

Per-pixel body-part segmentation with 29 classes (28 parts + background).

## Class Definitions

```
0: Background      10: Left_Sock         20: Right_Upper_Arm
1: Apparel         11: Left_Upper_Arm    21: Right_Upper_Leg
2: Eyeglass        12: Left_Upper_Leg    22: Torso
3: Face_Neck       13: Lower_Clothing    23: Upper_Clothing
4: Hair            14: Right_Foot        24: Lower_Lip
5: Left_Foot       15: Right_Hand        25: Upper_Lip
6: Left_Hand       16: Right_Lower_Arm   26: Lower_Teeth
7: Left_Lower_Arm  17: Right_Lower_Leg   27: Upper_Teeth
8: Left_Lower_Leg  18: Right_Shoe        28: Tongue
9: Left_Shoe       19: Right_Sock
```

## Model Zoo

Download checkpoints from [HuggingFace](https://huggingface.co/facebook/sapiens2)
and place them under `$SAPIENS_CHECKPOINT_ROOT` (default: `~/sapiens2_host`).

| Model | Checkpoint Path |
|-------|-----------------|
| Sapiens2-0.4B | `$SAPIENS_CHECKPOINT_ROOT/seg/sapiens2_0.4b_seg.safetensors` |
| Sapiens2-0.8B | `$SAPIENS_CHECKPOINT_ROOT/seg/sapiens2_0.8b_seg.safetensors` |
| Sapiens2-1B   | `$SAPIENS_CHECKPOINT_ROOT/seg/sapiens2_1b_seg.safetensors` |
| Sapiens2-5B   | `$SAPIENS_CHECKPOINT_ROOT/seg/sapiens2_5b_seg.safetensors` |

## Inference Guide

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/demo/seg.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — uncomment the model size you want to use
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 3 jobs/GPU on GPUs 0–7)

Outputs:
- Visualization images (color-coded class overlays)
- `.npy` files with raw class probabilities and foreground masks
  (used downstream by NORMAL, ALBEDO, POINTMAP)
