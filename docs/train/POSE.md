# Training Sapiens2 for Pose Estimation

How to train Sapiens2 for 308-keypoint pose estimation on your own dataset.

## 1. Data Preparation

Set `$DATA_ROOT` to your dataset root:
```bash
export DATA_ROOT=/path/to/your/pose_data
```

Sapiens2 uses the [308-keypoint Sociopticon format](../../sapiens/pose/configs/_base_/keypoints308.py) — body, dense face (274 kpts), hands, and feet. Your annotations should follow this layout.

Expected directory structure:
```
$DATA_ROOT/
├── images/                  # input RGB images
└── annotations/
    └── train.json           # per-image annotations
```

If your dataset uses a different keypoint format, you'll also need to either:
1. Convert your annotations to the 308-kpt format, **or**
2. Implement a new dataset class under `sapiens/pose/src/datasets/` and register it.

See `keypoints308_3po_dataset.py` for a reference implementation.

## 2. Configuration

Pick a model size from `sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/`:
- `sapiens2_0.4b_*.py`, `sapiens2_0.8b_*.py`, `sapiens2_1b_*.py`, `sapiens2_5b_*.py`

Edit your chosen config and set:
- `pretrained_checkpoint` — path to a pretrained backbone from [HuggingFace](https://huggingface.co/facebook/sapiens2)
  (e.g., `f"{_CHECKPOINT_ROOT}/pretrain/sapiens2_1b_pretrain.safetensors"`)
- `train_datasets` — point at your dataset class with `ann_file=f"{_DATA_ROOT}/annotations/train.json"`
- `num_keypoints` — only change if your skeleton differs from 308
- `num_iters`, `warmup_iters`, `save_every_iters` — schedule knobs

## 3. Launch Training

```bash
cd $SAPIENS_ROOT/sapiens/pose
./scripts/keypoints308/train/sapiens2_1b/node.sh
```

Open `node.sh` to adjust:
- `DEVICES` — which GPU IDs (default `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size
- `mode='multi-gpu'` — production mode; `mode='debug'` for single-GPU dry-run
- `LOAD_FROM` — checkpoint path to initialize weights from (e.g., to fine-tune from a released task checkpoint)
- `RESUME_FROM` — checkpoint to resume training from (continues epoch/iteration counter)

Outputs (checkpoints + logs) are written to:
```
Outputs/keypoints308/train/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768/node/<timestamp>/
```

For multi-node SLURM training, write a thin SLURM wrapper around `node.sh` (we don't ship one — clusters vary).
