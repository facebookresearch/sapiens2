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
class AlbedoVisualizer(nn.Module):
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

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        pred_albedos = logs["outputs"]
        pred_albedos = pred_albedos.detach().cpu()  # B x 3 x H x W
        gt_albedos = (
            data_batch["data_samples"]["gt_albedo"].detach().cpu()
        )  # B x 3 x H x W
        masks = data_batch["data_samples"]["mask"].detach().cpu()  # B x 1 x H x
        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W

        if pred_albedos.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_albedos = pred_albedos.float()

        pred_albedos = pred_albedos.cpu().detach().numpy()  ## B x 3 x H x W
        pred_albedos = pred_albedos.transpose((0, 2, 3, 1))  ## B x H x W x 3

        gt_albedos = gt_albedos.cpu().detach().numpy()  ## B x 3 x H x W
        gt_albedos = gt_albedos.transpose((0, 2, 3, 1))  ## B x H x W x 3

        batch_size = min(len(inputs), self.vis_max_samples)
        inputs = inputs[:batch_size]
        pred_albedos = pred_albedos[:batch_size]  ## B x 3 x H x W
        gt_albedos = gt_albedos[:batch_size]  ## B x 3 x H x W
        masks = masks[:batch_size]  ## B x 1 x H x W

        prefix = os.path.join(self.output_dir, "train")
        suffix = str(step).zfill(6)
        suffix += "_" + data_batch["data_samples"]["meta"]["img_path"][0].split("/")[
            -1
        ].replace(".png", "")
        vis_images = []

        for i, (input, gt_albedo, mask, pred_albedo) in enumerate(
            zip(inputs, gt_albedos, masks, pred_albedos)
        ):
            image = input.permute(1, 2, 0).cpu().numpy()  ## bgr image
            image = np.ascontiguousarray(image.copy())
            mask = mask[0].numpy() > 0  ## H x W

            if (
                pred_albedo.shape[0] != image.shape[0]
                or pred_albedo.shape[1] != image.shape[1]
            ):
                image = cv2.resize(
                    image,
                    (pred_albedo.shape[1], pred_albedo.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            vis_gt_albedo = (gt_albedo * 255).astype(np.uint8)  ## rgb
            vis_pred_albedo = (pred_albedo * 255).astype(np.uint8)  ## rgb

            vis_gt_albedo = cv2.cvtColor(vis_gt_albedo, cv2.COLOR_RGB2BGR)
            vis_pred_albedo = cv2.cvtColor(vis_pred_albedo, cv2.COLOR_RGB2BGR)

            vis_image = np.concatenate(
                [
                    image,
                    vis_gt_albedo,
                    vis_pred_albedo,
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
