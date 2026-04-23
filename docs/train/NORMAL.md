# Training Sapiens2 for Surface Normal Estimation

How to train Sapiens2 for per-pixel surface normal estimation on your own dataset.

## 1. Data Preparation

Set `$DATA_ROOT` to your dataset root:
```bash
export DATA_ROOT=/path/to/your/normal_data
```

Surface normals should be 3-channel (x, y, z) unit vectors per pixel, encoded
either as `.npy` arrays or RGB images (with the standard `n = 2*rgb - 1`
remapping).

Expected directory structure:
```
$DATA_ROOT/
├── images/                  # input RGB images
├── normals/                 # ground-truth normals (.npy or .png)
└── annotations/
    └── train.json           # image/normal pair index
```

See `normal_metasim_dataset.py`, `normal_render_people_body_dataset.py`,
or `normal_thuman_dataset.py` for reference dataset classes.

## 2. Configuration

Pick a model size from `sapiens/dense/configs/normal/metasim_render_people/`:
- `sapiens2_0.4b_*.py`, `sapiens2_0.8b_*.py`, `sapiens2_1b_*.py`, `sapiens2_5b_*.py`

Edit your chosen config and set:
- `pretrained_checkpoint` — path to a pretrained backbone from [HuggingFace](https://huggingface.co/facebook/sapiens2)
- `train_datasets` — point at your dataset class with `ann_file=f"{_DATA_ROOT}/annotations/train.json"`
- `num_iters`, `warmup_iters`, `save_every_iters` — schedule knobs

## 3. Launch Training

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/normal/train/sapiens2_1b/node.sh
```

Open `node.sh` to adjust:
- `DEVICES` — which GPU IDs (default `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size
- `mode='multi-gpu'` — production mode; `mode='debug'` for single-GPU dry-run
- `LOAD_FROM` — checkpoint path to initialize weights from
- `RESUME_FROM` — checkpoint to resume training from

Outputs (checkpoints + logs) are written to:
```
Outputs/normal/train/sapiens2_1b_normal_metasim_render_people-1024x768/node/<timestamp>/
```

For multi-node SLURM training, write a thin SLURM wrapper around `node.sh` (we don't ship one — clusters vary).
