# Sapiens2: Surface Normal Estimation

Per-pixel surface normal estimation. Predictions are 3-channel (x, y, z) unit
vectors in the camera coordinate frame.

## Model Zoo

Download checkpoints from [HuggingFace](https://huggingface.co/facebook/sapiens2)
and place them under `$SAPIENS_CHECKPOINT_ROOT` (default: `~/sapiens2_host`).

| Model | Checkpoint Path |
|-------|-----------------|
| Sapiens2-0.4B | `$SAPIENS_CHECKPOINT_ROOT/normal/sapiens2_0.4b_normal.safetensors` |
| Sapiens2-0.8B | `$SAPIENS_CHECKPOINT_ROOT/normal/sapiens2_0.8b_normal.safetensors` |
| Sapiens2-1B   | `$SAPIENS_CHECKPOINT_ROOT/normal/sapiens2_1b_normal.safetensors` |
| Sapiens2-5B   | `$SAPIENS_CHECKPOINT_ROOT/normal/sapiens2_5b_normal.safetensors` |

## Inference Guide

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/demo/normal.sh
```

Open the script and adjust:
- `INPUT` — path to your image directory
- `OUTPUT` — where to save visualizations
- `MODEL_NAME` — uncomment the model size you want to use
- `JOBS_PER_GPU`, `GPU_IDS` — parallelism (defaults: 3 jobs/GPU on GPUs 0–7)

For best results, run [body-part segmentation](SEG.md) first and pass the
foreground mask `.npy` to filter background pixels from the visualization.
