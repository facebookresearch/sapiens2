<p align="center">
  <img src="./docs/assets/sapiens2.gif" alt="Sapiens2" title="Sapiens2" width="500"/>
</p>

<p align="center">
   <h2 align="center">Scale. Semantics. Fidelity.</h2>
   <p align="center">
      <a href="https://rawalkhirodkar.github.io/"><strong>Rawal Khirodkar</strong></a>
      ·
      <a href="https://www.linkedin.com/in/hewen/"><strong>He Wen</strong></a>
      ·
      <a href="https://una-dinosauria.github.io/"><strong>Julieta Martinez</strong></a>
      ·
      <a href="https://www.linkedin.com/in/ydong142857/"><strong>Yuan Dong</strong></a>
      ·
      <a href="https://about.meta.com/realitylabs/"><strong>Su Zhaoen</strong></a>
      ·
      <a href="https://shunsukesaito.github.io/"><strong>Shunsuke Saito</strong></a>
   </p>
   <h3 align="center">ICLR 2026</h3>
</p>

<p align="center">
   <a href="https://arxiv.org/pdf/2604.21681">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
   </a>

   <a href='https://huggingface.co/collections/facebook/sapiens2'>
      <img src='https://img.shields.io/badge/HuggingFace-Hub-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF5500&color=orange' alt='HuggingFace Hub'>
   </a>
</p>

A family of high-resolution transformers pretrained on 1 billion human images, achieving state-of-the-art performance across diverse human-centric tasks — pose estimation, body-part segmentation, surface normals, and pointmaps.

## ⚡ Quick Start

Run a pretrained backbone forward pass — only `torch` and `safetensors` needed:

```python
import os
import torch
from safetensors.torch import load_file
from sapiens.backbones.standalone.sapiens2 import Sapiens2

# Build the model and load a pretrained checkpoint
model = Sapiens2(arch="sapiens2_1b", img_size=(1024, 768), patch_size=16).eval().cuda()  # img_size is (H, W)
ckpt = os.path.expanduser("~/sapiens2_host/pretrain/sapiens2_1b_pretrain.safetensors")
model.load_state_dict(load_file(ckpt))

# Forward pass on a single image (RGB; ImageNet normalization recommended)
x = torch.randn(1, 3, 1024, 768).cuda()
with torch.no_grad():
    features = model(x)[0]  # dense backbone features
```

The standalone files [`sapiens/backbones/standalone/sapiens2.py`](sapiens/backbones/standalone/sapiens2.py) and [`sapiens/backbones/standalone/sapiens.py`](sapiens/backbones/standalone/sapiens.py) (v1) are self-contained — copy either into your own project to use the backbone without the rest of this repo.

## 🧬 Model Card

| Model | Params | FLOPs | Embed dim | Layers | Heads |
|-------|--------|-------|-----------|--------|-------|
| Sapiens2-0.1B | 0.114 B |  0.342 T |  768 | 12 | 12 |
| Sapiens2-0.4B | 0.398 B |  1.260 T | 1024 | 24 | 16 |
| Sapiens2-0.8B | 0.818 B |  2.592 T | 1280 | 32 | 16 |
| Sapiens2-1B   | 1.462 B |  4.715 T | 1536 | 40 | 24 |
| Sapiens2-1B (4K) | 1.607 B |       — | 1536 | 40 | 24 |
| Sapiens2-5B   | 5.071 B | 15.722 T | 2432 | 56 | 32 |

All models use patch size 16 and are trained at 1024×768 (H×W) resolution, except Sapiens2-1B (4K) which is trained at 4096×3072 with `use_tokenizer=True`.

## 📦 Getting Started

**Clone the repository:**
```bash
git clone https://github.com/facebookresearch/sapiens2.git
cd sapiens2
export SAPIENS_ROOT=$(pwd)
```

**Install** (requires Python ≥3.12 and PyTorch ≥2.7):
```bash
pip install -e .
```

**Download checkpoints** from [MODEL_ZOO.md](docs/MODEL_ZOO.md). Place downloaded files under `$SAPIENS_CHECKPOINT_ROOT` (default: `~/sapiens2_host`):
```plaintext
sapiens2_host/
├── pretrain/
│   ├── sapiens2_{0.1b,0.4b,0.8b,1b,5b}_pretrain.safetensors
│   └── sapiens2_1b_4k_pretrain.safetensors
├── pose/
│   └── sapiens2_{0.4b,0.8b,1b,5b}_pose.safetensors
├── seg/
│   └── sapiens2_{0.4b,0.8b,1b,5b}_seg.safetensors
├── normal/
│   └── sapiens2_{0.4b,0.8b,1b,5b}_normal.safetensors
├── pointmap/
│   └── sapiens2_{0.4b,0.8b,1b,5b}_pointmap.safetensors
└── detector/
    └── rtmdet_m.pth          # bbox detector for pose (RTMDet)
```

## 🎯 Vision Tasks
| Task | Inference | Train |
|------|-----------|-------|
| Pose Estimation | [docs/POSE.md](docs/POSE.md) | [docs/train/POSE.md](docs/train/POSE.md) |
| Body-Part Segmentation | [docs/SEG.md](docs/SEG.md) | [docs/train/SEG.md](docs/train/SEG.md) |
| Surface Normal Estimation | [docs/NORMAL.md](docs/NORMAL.md) | [docs/train/NORMAL.md](docs/train/NORMAL.md) |
| Pointmap Estimation | [docs/POINTMAP.md](docs/POINTMAP.md) | [docs/train/POINTMAP.md](docs/train/POINTMAP.md) |

## ✨ Acknowledgements
We would like to acknowledge the contributions of [DINOv3](https://github.com/facebookresearch/dinov3), [OpenMMLab](https://github.com/open-mmlab), and [Accelerate](https://github.com/huggingface/accelerate), which this project benefits from.

## 🤝 Contributing
For questions or issues, please open an issue on GitHub. See [CONTRIBUTING](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## 📜 License
This project is licensed under the [Sapiens2 License](LICENSE.md).

## 📚 Citation
If you use Sapiens2 in your research, please consider citing us.
```bibtex
@article{khirodkarsapiens2,
  title={Sapiens2},
  author={Khirodkar, Rawal and Wen, He and Martinez, Julieta and Dong, Yuan and Su, Zhaoen and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2604.21681},
  year={2026}
}
```
