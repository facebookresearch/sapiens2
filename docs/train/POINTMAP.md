# Training Sapiens2 for Pointmap Estimation

How to train Sapiens2 for per-pixel 3D pointmap estimation on your own dataset.

## 1. Data Preparation

Set `$DATA_ROOT` to your dataset root:
```bash
export DATA_ROOT=/path/to/your/pointmap_data
```

A pointmap is a per-pixel (x, y, z) coordinate map in the camera frame —
typically derived from depth + camera intrinsics, or from rendered 3D meshes.

Expected directory structure:
```
$DATA_ROOT/
├── images/                  # input RGB images
├── pointmaps/               # ground-truth (H, W, 3) coordinate maps (.npy)
└── annotations/
    └── train.json           # image/pointmap pair index
```

See `pointmap_render_people_dataset.py` for a reference dataset class.

## 2. Configuration

Pick a model size from `sapiens/dense/configs/pointmap/render_people/`:
- `sapiens2_0.4b_*.py`, `sapiens2_0.8b_*.py`, `sapiens2_1b_*.py`, `sapiens2_5b_*.py`

Edit your chosen config and set:
- `pretrained_checkpoint` — path to a pretrained backbone from [HuggingFace](https://huggingface.co/facebook/sapiens2)
- `train_datasets` — point at your dataset class with `ann_file=f"{_DATA_ROOT}/annotations/train.json"`
- `num_iters`, `warmup_iters`, `save_every_iters` — schedule knobs

## 3. Launch Training

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/pointmap/train/sapiens2_1b/node.sh
```

Open `node.sh` to adjust:
- `DEVICES` — which GPU IDs (default `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size
- `mode='multi-gpu'` — production mode; `mode='debug'` for single-GPU dry-run
- `LOAD_FROM` — checkpoint path to initialize weights from
- `RESUME_FROM` — checkpoint to resume training from

Outputs (checkpoints + logs) are written to:
```
Outputs/pointmap/train/sapiens2_1b_pointmap_render_people-1024x768/node/<timestamp>/
```

For multi-node SLURM training, write a thin SLURM wrapper around `node.sh` (we don't ship one — clusters vary).
