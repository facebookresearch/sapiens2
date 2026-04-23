# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sapiens.registry import VISUALIZERS
from torch import nn


@VISUALIZERS.register_module()
class PointmapVisualizer(nn.Module):
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
        self.cmap = plt.get_cmap("turbo")
        self.error_cmap = plt.get_cmap("hot")

    def vis_point_map(self, point_map, mask=None):
        depth_map = point_map[:, :, 2]  ### x,y,z. z is the depth
        img = self.vis_depth_map(depth_map, mask=mask)
        return img

    def vis_depth_map(self, depth, mask=None, background_color=100):
        if mask is None:
            inverse_depth = 1 / depth
            inverse_depth_normalized = (inverse_depth - inverse_depth.min()) / (
                inverse_depth.max() - inverse_depth.min()
            )
            color_depth = (self.cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )

            ## convert RGB to BGR to save with cv2
            color_depth = color_depth[..., ::-1]
            return color_depth

        depth_foreground = depth[mask > 0]
        processed_depth = np.full(
            (mask.shape[0], mask.shape[1], 3), background_color, dtype=np.uint8
        )

        if len(depth_foreground) == 0:
            return processed_depth

        inverse_depth_foreground = 1 / depth_foreground

        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_inverse_depth = min(inverse_depth_foreground.max(), 1 / 0.1)
        min_inverse_depth = max(1 / 250, inverse_depth_foreground.min())
        inverse_depth_foreground_normalized = (
            inverse_depth_foreground - min_inverse_depth
        ) / (max_inverse_depth - min_inverse_depth)

        color_depth = (
            self.cmap(inverse_depth_foreground_normalized)[..., :3] * 255
        ).astype(np.uint8)
        processed_depth[mask] = color_depth

        ## convert RGB to BGR to save with cv2
        processed_depth = processed_depth[..., ::-1]

        return processed_depth

    def vis_normal_from_point_map(self, point_map, mask=None, kernel_size=7):
        depth_map = point_map[:, :, 2]  ### x,y,z. z is the depth

        if mask.sum() == 0:
            return np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

        depth_foreground = depth_map[mask > 0]
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)

        depth_normalized = np.full(mask.shape, np.inf)
        depth_normalized[mask > 0] = 1 - (
            (depth_map[mask > 0] - min_val) / (max_val - min_val)
        )

        grad_x = cv2.Sobel(
            depth_normalized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=kernel_size
        )
        grad_y = cv2.Sobel(
            depth_normalized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=kernel_size
        )
        normals = np.dstack((-grad_x, -grad_y, np.full(grad_x.shape, -1)))

        normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)
        normals_normalized = normals / (normals_mag + 1e-5)
        normal_vis = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)
        return normal_vis[:, :, ::-1]

    def vis_l1_error(self, gt_pointmap, pred_pointmap, mask=None, background_color=100):
        """Visualize L1 error between ground truth and predicted pointmaps."""
        if mask is None:
            mask = np.ones_like(gt_pointmap[:, :, 0], dtype=bool)

        error_map = np.full(
            (mask.shape[0], mask.shape[1], 3), background_color, dtype=np.uint8
        )

        # Calculate L1 error for valid points
        l1_error = np.abs(gt_pointmap - pred_pointmap)  # H x W x 3
        l1_error = np.mean(l1_error, axis=2)  # Average across XYZ dimensions, H x W

        if np.sum(mask) > 0:
            error_foreground = l1_error[mask]

            # Normalize error for visualization
            error_normalized = (error_foreground - error_foreground.min()) / (
                error_foreground.max() - error_foreground.min() + 1e-6
            )

            # Convert to color using hot colormap
            error_colored = (self.error_cmap(error_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            error_map[mask] = error_colored

            # Convert to BGR for OpenCV
            error_map = error_map[..., ::-1]

        return error_map

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        (pred_pointmaps, _) = logs["outputs"]
        pred_pointmaps = pred_pointmaps.detach().cpu()  # B x 3 x H x W
        gt_pointmaps = (
            data_batch["data_samples"]["gt_pointmap"].detach().cpu()
        )  # B x 3 x H x
        masks = data_batch["data_samples"]["mask"].detach().cpu()  # B x 1 x H x
        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W

        if pred_pointmaps.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_pointmaps = pred_pointmaps.float()

        pred_pointmaps = pred_pointmaps.cpu().detach().numpy()  ## B x 3 x H x W
        pred_pointmaps = pred_pointmaps.transpose((0, 2, 3, 1))  ## B x H x W x 3
        batch_size = min(len(inputs), self.vis_max_samples)

        inputs = inputs[:batch_size]
        pred_pointmaps = pred_pointmaps[:batch_size]  ## B x 3 x H x W
        gt_pointmaps = gt_pointmaps[:batch_size]  ## B x 3 x H x W
        masks = masks[:batch_size]  ## B x 1 x H x W

        prefix = os.path.join(self.output_dir, "train")
        suffix = str(step).zfill(6)
        suffix += "_" + data_batch["data_samples"]["meta"]["img_path"][0].split("/")[
            -1
        ].replace(".png", "")
        vis_images = []

        for i, (input, gt_pointmap, mask, pred_pointmap) in enumerate(
            zip(inputs, gt_pointmaps, masks, pred_pointmaps)
        ):
            image = input.permute(1, 2, 0).cpu().numpy()  ## bgr image
            image = np.ascontiguousarray(image.copy())

            gt_pointmap = gt_pointmap.numpy()  ## 3 x H x W
            gt_pointmap = gt_pointmap.transpose((1, 2, 0))  ## H x W x 3
            mask = mask[0].numpy() > 0  ## H x W

            ## resize predpoint to image size
            if (
                pred_pointmap.shape[0] != image.shape[0]
                or pred_pointmap.shape[1] != image.shape[1]
            ):
                image = cv2.resize(
                    image,
                    (pred_pointmap.shape[1], pred_pointmap.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            vis_gt_pointmap = self.vis_point_map(gt_pointmap, mask)
            vis_pred_pointmap = self.vis_point_map(pred_pointmap, mask)

            vis_gt_normal = self.vis_normal_from_point_map(gt_pointmap, mask)
            vis_pred_normal = self.vis_normal_from_point_map(pred_pointmap, mask)

            vis_error = self.vis_l1_error(gt_pointmap, pred_pointmap, mask)

            vis_image = np.concatenate(
                [
                    image,
                    vis_gt_pointmap,
                    vis_gt_normal,
                    vis_pred_pointmap,
                    vis_pred_normal,
                    vis_error,
                ],
                axis=1,
            )
            vis_image = cv2.resize(
                vis_image,
                (6 * self.vis_image_width, self.vis_image_height),
                interpolation=cv2.INTER_AREA,
            )
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = "{}_{}.jpg".format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
