# Training Sapiens2 for Body-Part Segmentation

How to train Sapiens2 for 29-class body-part segmentation on your own dataset.

## 1. Data Preparation

Set `$DATA_ROOT` to your dataset root:
```bash
export DATA_ROOT=/path/to/your/seg_data
```

Sapiens2 uses [29 body-part classes](../SEG.md#class-definitions) (28 parts + background). Your masks should use the same class indices, or you'll need a class mapping (see `INTERNAL_MAPPING_32_to_29` in `seg_utils.py` for an example).

Expected directory structure:
```
$DATA_ROOT/
├── images/                  # input RGB images
├── masks/                   # per-pixel class indices (0–28)
└── annotations/
    └── train.json           # image/mask pair index
```

If your dataset uses a different number of classes, you'll need to either:
1. Map your classes to the 29-class taxonomy, **or**
2. Change `num_classes` in the config and retrain the segmentation head.

See `seg_dome_dataset.py` and `seg_internal_dataset.py` for reference dataset classes.

## 2. Configuration

Pick a model size from `sapiens/dense/configs/seg/shutterstock_goliath/`:
- `sapiens2_0.4b_*.py`, `sapiens2_0.8b_*.py`, `sapiens2_1b_*.py`, `sapiens2_5b_*.py`

Edit your chosen config and set:
- `pretrained_checkpoint` — path to a pretrained backbone from [HuggingFace](https://huggingface.co/facebook/sapiens2)
- `train_datasets` — point at your dataset class with `ann_file=f"{_DATA_ROOT}/annotations/train.json"`
- `num_classes` — number of segmentation classes (default 29, change to match your data)
- `num_iters`, `warmup_iters`, `save_every_iters` — schedule knobs

## 3. Launch Training

```bash
cd $SAPIENS_ROOT/sapiens/dense
./scripts/seg/train/sapiens2_1b/node.sh
```

Open `node.sh` to adjust:
- `DEVICES` — which GPU IDs (default `0,1,2,3,4,5,6,7`)
- `TRAIN_BATCH_SIZE_PER_GPU` — per-GPU batch size
- `mode='multi-gpu'` — production mode; `mode='debug'` for single-GPU dry-run
- `LOAD_FROM` — checkpoint path to initialize weights from (e.g., to fine-tune from a released task checkpoint)
- `RESUME_FROM` — checkpoint to resume training from

Outputs (checkpoints + logs) are written to:
```
Outputs/seg/train/sapiens2_1b_seg_shutterstock_goliath-1024x768/node/<timestamp>/
```

For multi-node SLURM training, write a thin SLURM wrapper around `node.sh` (we don't ship one — clusters vary).
