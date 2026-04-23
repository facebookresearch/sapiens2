# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from sapiens.registry import VISUALIZERS
from torch import nn


@VISUALIZERS.register_module()
class NormalVisualizer(nn.Module):
    def __init__(
        self,
        output_dir: str,
        vis_interval: int = 100,
        vis_max_samples: int = 4,
        vis_image_width: int = 384,
        vis_image_height: int = 512,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_max_samples = vis_max_samples
        self.vis_interval = vis_interval
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height

    def vis_normal(self, normal_map, mask=None):
        normal_map[mask == 0] = np.nan
        normal_map_vis = (((normal_map + 1) / 2) * 255).astype(np.uint8)
        ## bgr to rgb
        normal_map_vis = normal_map_vis[:, :, ::-1]
        return normal_map_vis

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        pred_normals = logs["outputs"]
        pred_normals = pred_normals.detach().cpu()  # B x 3 x H x W
        gt_normals = (
            data_batch["data_samples"]["gt_normal"].detach().cpu()
        )  # B x 3 x H x W
        masks = data_batch["data_samples"]["mask"].detach().cpu()  # B x 1 x H x
        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W

        if pred_normals.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_normals = pred_normals.float()

        pred_normals = pred_normals.cpu().detach().numpy()  ## B x 3 x H x W
        pred_normals = pred_normals.transpose((0, 2, 3, 1))  ## B x H x W x 3
        batch_size = min(len(inputs), self.vis_max_samples)

        inputs = inputs[:batch_size]
        pred_normals = pred_normals[:batch_size]  ## B x 3 x H x W
        gt_normals = gt_normals[:batch_size]  ## B x 3 x H x W
        masks = masks[:batch_size]  ## B x 1 x H x W

        prefix = os.path.join(self.output_dir, "train")
        suffix = str(step).zfill(6)
        suffix += "_" + data_batch["data_samples"]["meta"]["img_path"][0].split("/")[
            -1
        ].replace(".png", "")
        vis_images = []

        for i, (input, gt_normal, mask, pred_normal) in enumerate(
            zip(inputs, gt_normals, masks, pred_normals)
        ):
            image = input.permute(1, 2, 0).cpu().numpy()  ## bgr image
            image = np.ascontiguousarray(image.copy())

            gt_normal = gt_normal.numpy()  ## 3 x H x W
            gt_normal = gt_normal.transpose((1, 2, 0))  ## H x W x 3
            mask = mask[0].numpy() > 0  ## H x W

            if (
                pred_normal.shape[0] != image.shape[0]
                or pred_normal.shape[1] != image.shape[1]
            ):
                image = cv2.resize(
                    image,
                    (pred_normal.shape[1], pred_normal.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            vis_gt_normal = self.vis_normal(gt_normal, mask)
            vis_pred_normal = self.vis_normal(pred_normal, mask)

            vis_image = np.concatenate(
                [
                    image,
                    vis_gt_normal,
                    vis_pred_normal,
                ],
                axis=1,
            )
            vis_image = cv2.resize(
                vis_image,
                (3 * self.vis_image_width, self.vis_image_height),
                interpolation=cv2.INTER_AREA,
            )
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = "{}_{}.jpg".format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
