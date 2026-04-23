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
from sapiens.dense.visualizers import SegVisualizer
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument(
        "--class_palette_type", default="dome29", help="Color palette for 29 classes"
    )
    parser.add_argument(
        "--save_pred", action="store_true", help="Save prediction to file"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
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

    class_palette_type = args.class_palette_type
    visualizer = SegVisualizer(class_palette_type=class_palette_type, with_labels=False)

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs = data["inputs"]

        ## pointmap is 1 x 3 x H x W, scale is 1 x 1
        with torch.no_grad():
            seg_logits = model(inputs)

        # resize prediction to image size
        seg_logits = F.interpolate(
            seg_logits, size=image.shape[:2], mode="bilinear"
        )  ## 1 x C x H x W
        pred_labels = seg_logits.argmax(dim=1).cpu().numpy()  ## 1 x H x W
        pred_labels = pred_labels.squeeze(0)  ## H x W

        vis_seg = visualizer._visualize_segmentation(image, pred_labels)
        vis_image = np.concatenate([image, vis_seg], axis=1)
        base_path = os.path.join(args.output, image_name.rsplit(".")[0])
        cv2.imwrite(f"{base_path}.{image_name.rsplit('.')[1]}", vis_image)

        if args.save_pred:
            np.save(f"{base_path}_seg.npy", pred_labels)


if __name__ == "__main__":
    main()
