# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from sapiens.registry import VISUALIZERS
from torch import nn


@VISUALIZERS.register_module()
class BaseVisualizer(nn.Module):
    def __init__(
        self,
        output_dir: str,
        vis_interval: int = 100,
        vis_max_samples: int = 16,
        vis_downsample: int = 2,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_max_samples = vis_max_samples
        self.vis_interval = vis_interval
        self.vis_downsample = vis_downsample

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        images = data_batch["data_samples"]["image"].detach().cpu()
        outputs = logs["outputs"].detach().cpu()  # (B, C, H, W)

        if outputs.dtype == torch.bfloat16:
            images = images.float()
            outputs = outputs.float()

        batch_size = min(len(images), self.vis_max_samples)

        save_images = []

        for i in range(batch_size):
            gt_image = images[i].permute(1, 2, 0).cpu().numpy() * 255
            pred_image = outputs[i].permute(1, 2, 0).cpu().numpy() * 255

            gt_image = np.clip(gt_image, 0, 255).astype(np.uint8)
            pred_image = np.clip(pred_image, 0, 255).astype(np.uint8)

            image_height, image_width = gt_image.shape[:2]

            if self.vis_downsample > 1:
                image_height = int(image_height / self.vis_downsample)
                image_width = int(image_width / self.vis_downsample)

                gt_image = cv2.resize(
                    gt_image,
                    (image_width, image_height),
                    interpolation=cv2.INTER_AREA,
                )
                pred_image = cv2.resize(
                    pred_image,
                    (image_width, image_height),
                    interpolation=cv2.INTER_AREA,
                )

            save_image = np.concatenate([gt_image, pred_image], axis=1)
            save_images.append(save_image)

        out_file = self.output_dir / f"{step:06d}.jpg"
        image_height, image_width = save_images[0].shape[:2]
        cols = int(math.ceil(math.sqrt(batch_size)))
        rows = int(math.ceil(batch_size / cols))

        canvas_height = rows * image_height
        canvas_width = cols * image_width

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for idx, image in enumerate(save_images):
            row = idx // cols
            col = idx % cols
            canvas[
                row * image_height : (row + 1) * image_height,
                col * image_width : (col + 1) * image_width,
            ] = image

        ## downsample canvas by 2x
        canvas = cv2.resize(
            canvas,
            (canvas_width, canvas_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(out_file, canvas)

        return
