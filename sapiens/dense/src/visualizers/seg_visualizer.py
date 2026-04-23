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

from ..datasets import DOME_CLASSES_29


@VISUALIZERS.register_module()
class SegVisualizer(nn.Module):
    def __init__(
        self,
        output_dir: str = None,
        vis_interval: int = 100,
        vis_max_samples: int = 4,
        vis_image_width: int = 384,
        vis_image_height: int = 512,
        class_palette_type="dome29",
        overlay_opacity: float = 0.5,  # 0..1; 1 = only mask colors
        with_labels: bool = True,
    ):
        super().__init__()
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vis_max_samples = vis_max_samples
        self.vis_interval = vis_interval
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height
        self.class_palette_type = class_palette_type
        self.overlay_opacity = float(np.clip(overlay_opacity, 0.0, 1.0))
        self.with_labels = with_labels
        self.class_palette = None
        self.class_names = {}

        if self.class_palette_type == "dome29":
            self.class_palette = self._build_palette_from_dict(DOME_CLASSES_29)
            self.class_names = {
                cid: meta.get("name", f"class_{cid}")
                for cid, meta in DOME_CLASSES_29.items()
            }
        else:
            self.class_palette = (np.random.rand(256, 3) * 255).astype(np.uint8)

    def _build_palette_from_dict(self, class_dict) -> np.ndarray:
        max_id = max(int(k) for k in class_dict.keys())
        pal = np.zeros((max(max_id + 1, 256), 3), dtype=np.uint8)
        for cid, meta in class_dict.items():
            col = meta.get("color", [0, 0, 0])
            pal[int(cid)] = np.array(col, dtype=np.uint8)
        return pal  # RGB format

    def _get_center_loc(self, mask: np.ndarray):
        """
        Finds the center of the largest contour in a binary mask.
        This is a robust method using moments.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def _draw_labels(self, image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
        """Draws class labels on the image at the center of each segment."""
        unique_labels = np.unique(label_map)
        for class_id in unique_labels:
            if class_id == 0 or class_id not in self.class_names:
                continue  # Skip background or unknown classes

            class_name = self.class_names[class_id]
            mask = (label_map == class_id).astype(np.uint8)
            loc = self._get_center_loc(mask)
            if loc is None:
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (75 / scale)
            fontColor = (255, 255, 255)  # White text
            thickness = 1
            rectangleThickness = 1

            (label_width, label_height), baseline = cv2.getTextSize(
                class_name, font, fontScale, thickness
            )

            x, y = loc
            x = max(x - label_width // 2, 0)
            y_text = y + label_height // 2
            rect_start_pt = (x, y - label_height // 2 - baseline)
            rect_end_pt = (x + label_width, y + label_height // 2 + baseline)
            class_color_rgb = self.class_palette[class_id]
            class_color_bgr = tuple(int(c) for c in class_color_rgb[::-1])
            cv2.rectangle(image, rect_start_pt, rect_end_pt, class_color_bgr, -1)
            cv2.rectangle(
                image, rect_start_pt, rect_end_pt, (0, 0, 0), rectangleThickness
            )
            cv2.putText(
                image, class_name, (x, y_text), font, fontScale, fontColor, thickness
            )
        return image

    def _visualize_segmentation(
        self, image_bgr: np.ndarray, label_map: np.ndarray
    ) -> np.ndarray:
        if image_bgr.dtype != np.uint8:
            raise ValueError("Input image must be uint8 for visualization.")
        palette_bgr = self.class_palette[:, ::-1]
        color_mask = palette_bgr[label_map]

        if self.with_labels:
            color_mask = self._draw_labels(color_mask, label_map)

        overlay = cv2.addWeighted(
            image_bgr,
            1 - self.overlay_opacity,
            color_mask,
            self.overlay_opacity,
            0,
        )

        return overlay

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W
        pred_logits = logs["outputs"].detach().cpu()  # B x num_classes x H x W
        gt_labels = (data_batch["data_samples"]["gt_seg"].detach().cpu()).squeeze(
            dim=1
        )  ## B x H x W;

        if pred_logits.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_logits = pred_logits.float()

        pred_labels = pred_logits.argmax(dim=1)  ## B x H x W
        batch_size = min(len(inputs), self.vis_max_samples)

        inputs = inputs[:batch_size]
        gt_labels = gt_labels[:batch_size]  ## B x 1 x H x W
        pred_labels = pred_labels[:batch_size]  ## B x H x W

        prefix = os.path.join(self.output_dir, "train")
        suffix = str(step).zfill(6)
        vis_images = []

        for i, (input, gt_label, pred_label) in enumerate(
            zip(inputs, gt_labels, pred_labels)
        ):
            image = input.permute(1, 2, 0).cpu().numpy()  ## bgr image
            image = np.ascontiguousarray(image.copy()).astype(np.uint8)

            gt_label = gt_label.numpy().astype(np.uint8)  ## H x W
            pred_label = pred_label.numpy().astype(np.uint8)  ## H x W

            vis_gt_seg = self._visualize_segmentation(image, gt_label)
            vis_pred_seg = self._visualize_segmentation(image, pred_label)

            vis_image = np.concatenate(
                [
                    image,
                    vis_gt_seg,
                    vis_pred_seg,
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
        grid_out_file = "{}_{}.jpg".format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
