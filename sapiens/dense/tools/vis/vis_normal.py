# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sapiens.dense.models import init_model
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument(
        "--no-black-background",
        "--no_black_background",
        action="store_true",
        help="No black background",
    )
    parser.add_argument(
        "--seg_dir", "--seg-dir", default=None, help="Path to segmentation dir"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="If provided, do not save .npy prediction files",
    )

    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    # Get image list
    if os.path.isdir(args.input):
        input_dir = args.input
        image_names = [
            name
            for name in sorted(os.listdir(input_dir))
            if name.endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]
        input_dir = os.path.dirname(image_paths[0])

    seg_dir = args.seg_dir

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        mask_path = os.path.join(
            seg_dir,
            image_name.replace(".png", ".npy")
            .replace(".jpg", ".npy")
            .replace(".jpeg", ".npy"),
        )

        mask_path_candidates = [
            mask_path,  # npy
            mask_path.replace(".npy", "_seg.npy"),  # npy, seg probs
            os.path.join(seg_dir, image_name),  # png or jpg
        ]

        mask = np.ones_like(image[:, :, 0], dtype=bool)
        for mask_path in mask_path_candidates:
            if not os.path.exists(mask_path):
                continue
            if mask_path.endswith("_seg.npy"):
                mask = np.load(mask_path)  ## H x W, float; class labels
                mask = mask > 0  ## skip the bg class
            elif mask_path.endswith(".npy"):
                mask = np.load(mask_path)  ## H x W, boolean
            else:
                mask = cv2.imread(mask_path)[:, :, 0]  ## H x W, uint8
                mask = mask > 0
            break

        ##------------------------------------------
        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs, data_samples = data["inputs"], data["data_samples"]

        with torch.no_grad():
            normal = model(inputs)  # normal is 1 x 3 x H x W
            normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(
                min=1e-8
            )  # normalize to unit length

        # ------------------------------------------
        pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"]["padding_size"]
        normal = normal[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]

        normal = F.interpolate(
            normal,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        normal = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3

        if not args.no_save_predictions:
            base_path = os.path.join(args.output, image_name.rsplit(".")[0])
            np.save(f"{base_path}.npy", normal)

        normal[mask == 0] = -1
        normal_vis = ((normal + 1) / 2 * 255).astype(np.uint8)
        normal_vis = normal_vis[:, :, ::-1]

        if args.no_black_background:
            normal_vis[mask == 0] = image[mask == 0]

        vis_image = np.concatenate([image, normal_vis], axis=1)
        save_path = os.path.join(args.output, image_name)
        cv2.imwrite(save_path, vis_image)


if __name__ == "__main__":
    main()
