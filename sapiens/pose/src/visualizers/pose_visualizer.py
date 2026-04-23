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
import torchvision
from sapiens.registry import VISUALIZERS
from torch import nn

from ..datasets.utils import parse_pose_metainfo


@VISUALIZERS.register_module()
class PoseVisualizer(nn.Module):
    def __init__(
        self,
        output_dir: str,
        vis_interval: int = 100,
        vis_max_samples: int = 4,
        vis_image_width: int = 384,
        vis_image_height: int = 512,
        num_keypoints: int = 308,
        scale: int = 4,
        line_width: int = 4,
        radius: int = 4,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_max_samples = vis_max_samples
        self.vis_interval = vis_interval
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height
        self.num_keypoints = num_keypoints
        self.scale = scale
        self.line_width = line_width
        self.radius = radius

        if self.num_keypoints == 308:
            self.dataset_meta = parse_pose_metainfo(
                dict(from_file="configs/_base_/keypoints308.py")
            )

        self.bbox_color = self.dataset_meta.get("bbox_colors", "green")
        self.kpt_color = self.dataset_meta.get("keypoint_colors")
        self.link_color = self.dataset_meta.get("skeleton_link_colors")
        self.skeleton = self.dataset_meta.get("skeleton_links")

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        pred_heatmaps = logs["outputs"]
        pred_heatmaps = pred_heatmaps.detach().cpu()  # B x K x H x W

        gt_heatmaps = (
            data_batch["data_samples"]["heatmaps"].detach().cpu()
        )  # B x K x H x W

        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W

        if pred_heatmaps.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_heatmaps = pred_heatmaps.float()

        pred_heatmaps = pred_heatmaps.cpu().detach().numpy()  ## B x K x H x W
        gt_heatmaps = gt_heatmaps.cpu().detach().numpy()  ## B x K x H x W
        target_weights = (
            data_batch["data_samples"]["keypoint_weights"].squeeze(dim=1).cpu().numpy()
        )  ## B x K

        batch_size = min(len(inputs), self.vis_max_samples)
        inputs = inputs[:batch_size]
        pred_heatmaps = pred_heatmaps[:batch_size]  ## B x K x H x W
        gt_heatmaps = gt_heatmaps[:batch_size]  ## B x K x H x W
        target_weights = target_weights[:batch_size]  ## B x K

        kps_vis_dir = os.path.join(self.output_dir, "kps")
        heatmap_vis_dir = os.path.join(self.output_dir, "heatmap")

        if not os.path.exists(kps_vis_dir):
            os.makedirs(kps_vis_dir, exist_ok=True)

        if not os.path.exists(heatmap_vis_dir):
            os.makedirs(heatmap_vis_dir, exist_ok=True)

        kps_prefix = os.path.join(kps_vis_dir, "train")
        heatmap_prefix = os.path.join(heatmap_vis_dir, "train")
        suffix = str(step).zfill(6)

        original_image = inputs / 255.0  ## B x 3 x H x W

        ## heatmap vis for only first 17 kps
        self.save_batch_heatmaps(
            original_image,
            gt_heatmaps[:, :17],
            "{}_{}_hm_gt.jpg".format(heatmap_prefix, suffix),
            normalize=False,
            scale=self.scale,
            is_rgb=False,
        )
        self.save_batch_heatmaps(
            original_image,
            pred_heatmaps[:, :17],
            "{}_{}_hm_pred.jpg".format(heatmap_prefix, suffix),
            normalize=False,
            scale=self.scale,
            is_rgb=False,
        )
        self.save_batch_image_with_joints(
            255 * original_image,
            gt_heatmaps,
            target_weights,
            "{}_{}_gt.jpg".format(kps_prefix, suffix),
            scale=self.scale,
            is_rgb=False,
        )
        self.save_batch_image_with_joints(
            255 * original_image,
            pred_heatmaps,
            np.ones_like(target_weights),
            "{}_{}_pred.jpg".format(kps_prefix, suffix),
            scale=self.scale,
            is_rgb=False,
        )
        return

    def save_batch_heatmaps(
        self,
        batch_image,
        batch_heatmaps,
        file_name,
        normalize=True,
        scale=4,
        is_rgb=True,
        max_num_joints=17,
    ):
        """
        batch_image: [batch_size, channel, height, width]
        batch_heatmaps: ['batch_size, num_joints, height, width]
        file_name: saved file name
        """
        ## normalize image
        if normalize:
            batch_image = batch_image.clone()
            min_val = float(batch_image.min())
            max_val = float(batch_image.max())

            batch_image.add_(-min_val).div_(max_val - min_val + 1e-5)

        ## check if type of batch_heatmaps is numpy.ndarray
        if isinstance(batch_heatmaps, np.ndarray):
            preds, maxvals = get_max_preds(batch_heatmaps)
            batch_heatmaps = torch.from_numpy(batch_heatmaps)
        else:
            preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

        preds = preds * scale  ## scale to original image size

        batch_size = batch_heatmaps.size(0)
        num_joints = batch_heatmaps.size(1)
        heatmap_height = int(batch_heatmaps.size(2) * scale)
        heatmap_width = int(batch_heatmaps.size(3) * scale)

        num_joints = min(max_num_joints, num_joints)

        grid_image = np.zeros(
            (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
            dtype=np.uint8,
        )

        body_joint_order = range(max_num_joints)

        for i in range(batch_size):
            image = (
                batch_image[i]
                .mul(255)
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

            if is_rgb == True:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

            height_begin = heatmap_height * i
            height_end = heatmap_height * (i + 1)
            for j in range(num_joints):
                joint_index = body_joint_order[j]

                cv2.circle(
                    resized_image,
                    (int(preds[i][joint_index][0]), int(preds[i][joint_index][1])),
                    1,
                    [0, 0, 255],
                    1,
                )
                heatmap = heatmaps[joint_index, :, :]
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                colored_heatmap = cv2.resize(
                    colored_heatmap, (int(heatmap_width), int(heatmap_height))
                )
                masked_image = colored_heatmap * 0.7 + resized_image * 0.3
                cv2.circle(
                    masked_image,
                    (int(preds[i][joint_index][0]), int(preds[i][joint_index][1])),
                    1,
                    [0, 0, 255],
                    1,
                )

                width_begin = heatmap_width * (j + 1)
                width_end = heatmap_width * (j + 2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = (
                    masked_image
                )

            grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

        ## resize
        target_height = batch_size * self.vis_image_height
        target_width = (num_joints + 1) * self.vis_image_width
        grid_image = cv2.resize(grid_image, (target_width, target_height))

        cv2.imwrite(file_name, grid_image)
        return

    def save_batch_image_with_joints(
        self,
        batch_image,
        batch_heatmaps,
        batch_target_weight,
        file_name,
        is_rgb=True,
        scale=4,
        nrow=8,
        padding=2,
    ):
        """
        batch_image: [batch_size, channel, height, width]
        batch_joints: [batch_size, num_joints, 3],
        batch_joints_vis: [batch_size, num_joints, 1],
        }
        """

        B, C, H, W = batch_image.size()
        num_joints = batch_heatmaps.shape[1]

        ## check if type of batch_heatmaps is numpy.ndarray
        if isinstance(batch_heatmaps, np.ndarray):
            batch_joints, batch_scores = get_max_preds(batch_heatmaps)
        else:
            batch_joints, batch_scores = get_max_preds(
                batch_heatmaps.detach().cpu().numpy()
            )

        batch_joints = (
            batch_joints * scale
        )  ## 4 is the ratio of output heatmap and input image

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, num_joints)  ## B x 17

        grid = []

        for i in range(B):
            image = (
                batch_image[i].permute(1, 2, 0).cpu().numpy()
            )  # image_size x image_size x BGR. if is_rgb is False.
            image = image.copy()
            kps = batch_joints[i]  ## 17 x 2
            kps_vis = batch_target_weight[i]
            kps_score = batch_scores[i].reshape(-1)

            if is_rgb == False:
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )  # convert bgr to rgb image

            kp_vis_image = self.draw_instance_kpts(
                image,
                keypoints=[kps],
                keypoints_visible=[kps_vis],
                keypoint_scores=[kps_score],
                radius=self.radius,
                thickness=self.line_width,
                kpt_thr=0.3,
                skeleton=self.skeleton,
                kpt_color=self.kpt_color,
                link_color=self.link_color,
            )  ## H, W, C, rgb image

            kp_vis_image = cv2.cvtColor(
                kp_vis_image, cv2.COLOR_RGB2BGR
            )  ## convert rgb to bgr image

            kp_vis_image = kp_vis_image.transpose((2, 0, 1)).astype(np.float32)
            kp_vis_image = torch.from_numpy(kp_vis_image.copy())
            grid.append(kp_vis_image)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()

        ## resize
        target_height = self.vis_image_height
        target_width = ndarr.shape[1] * target_height // ndarr.shape[0]
        ndarr = cv2.resize(ndarr, (target_width, target_height))

        cv2.imwrite(file_name, ndarr)
        return

    def draw_instance_kpts(
        self,
        image: np.ndarray,  # RGB uint8 H,W,3
        keypoints,  # list[(J,2)]
        keypoints_visible,  # list[(J,), {0/1}]
        keypoint_scores,  # list[(J,)]
        *,
        radius: int = 4,
        thickness: int = -1,
        color=(255, 0, 0),
        kpt_thr: float = 0.3,
        skeleton: list | None = None,  # [(i,j)]
        kpt_color: list | tuple | np.ndarray | None = None,
        link_color: list | tuple | np.ndarray | None = None,
        show_kpt_idx: bool = False,
    ) -> np.ndarray:
        img = image.copy()
        H, W = img.shape[:2]

        # defaults
        if skeleton is None:
            skeleton = []  # points only
        if kpt_color is None:
            kpt_color = color
        if link_color is None:
            link_color = (0, 255, 0)

        # robust color normalization: supports tuple, list-of-tuples, np.ndarray (N,3) or (3,)
        def _as_color_list(c, n):
            # torch -> numpy
            if hasattr(c, "detach"):
                c = c.detach().cpu().numpy()
            # numpy -> array
            if isinstance(c, np.ndarray):
                if c.ndim == 2 and c.shape[1] == 3:  # (N,3) palette
                    return [tuple(int(v) for v in row) for row in c.tolist()]
                if c.size == 3:  # single (3,)
                    return [tuple(int(v) for v in c.tolist())] * max(1, n)
            # python containers
            if isinstance(c, (list, tuple)):
                if n and len(c) == n and isinstance(c[0], (list, tuple, np.ndarray)):
                    out = []
                    for cc in c:
                        cc = np.asarray(cc).reshape(-1)
                        assert cc.size == 3, "Each color must be length-3"
                        out.append(tuple(int(v) for v in cc.tolist()))
                    return out
                # single triplet
                c_arr = np.asarray(c).reshape(-1)
                if c_arr.size == 3:
                    return [tuple(int(v) for v in c_arr.tolist())] * max(1, n)
            # fallback: red
            return [(255, 0, 0)] * max(1, n)

        J = keypoints[0].shape[0] if keypoints else 0
        kpt_colors = _as_color_list(kpt_color, J)
        link_colors = _as_color_list(link_color, len(skeleton))

        def in_bounds(x, y):
            return 0 <= x < W and 0 <= y < H

        for kpts, vis, score in zip(keypoints, keypoints_visible, keypoint_scores):
            kpts = np.asarray(kpts, float)
            vis = np.asarray(vis).reshape(-1).astype(bool)
            score = np.asarray(score).reshape(-1)

            # links (draw in RGB; NO channel flip)
            for lk, (i, j) in enumerate(skeleton):
                if i >= len(kpts) or j >= len(kpts):
                    continue
                if not (vis[i] and vis[j]):
                    continue
                if score[i] < kpt_thr or score[j] < kpt_thr:
                    continue

                x1, y1 = map(int, np.round(kpts[i]))
                x2, y2 = map(int, np.round(kpts[j]))
                if not (in_bounds(x1, y1) and in_bounds(x2, y2)):
                    continue

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    link_colors[lk % len(link_colors)],
                    thickness=max(1, self.line_width),
                    lineType=cv2.LINE_AA,
                )

            # points
            for j_idx, (xy, v, s) in enumerate(zip(kpts, vis, score)):
                if not v or s < kpt_thr:
                    continue
                x, y = map(int, np.round(xy))
                if not in_bounds(x, y):
                    continue

                c = kpt_colors[min(j_idx, len(kpt_colors) - 1)]
                cv2.circle(
                    img, (x, y), radius, c, thickness=thickness, lineType=cv2.LINE_AA
                )
                if show_kpt_idx:
                    cv2.putText(
                        img,
                        str(j_idx),
                        (x + radius, y - radius),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        c,
                        1,
                        cv2.LINE_AA,
                    )
        return img


###------------------helpers-----------------------
def batch_unnormalize_image(
    images, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
):
    normalize = transforms.Normalize(mean=mean, std=std)
    images[:, 0, :, :] = (images[:, 0, :, :] * normalize.std[0]) + normalize.mean[0]
    images[:, 1, :, :] = (images[:, 1, :, :] * normalize.std[1]) + normalize.mean[1]
    images[:, 2, :, :] = (images[:, 2, :, :] * normalize.std[2]) + normalize.mean[2]
    return images


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), (
        "batch_heatmaps should be numpy.ndarray"
    )
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)  ## B x 17
    maxvals = np.amax(heatmaps_reshaped, 2)  ## B x 17

    maxvals = maxvals.reshape((batch_size, num_joints, 1))  ## B x 17 x 1
    idx = idx.reshape((batch_size, num_joints, 1))  ## B x 17 x 1

    preds = np.tile(idx, (1, 1, 2)).astype(
        np.float32
    )  ## B x 17 x 2, like repeat in pytorch

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
