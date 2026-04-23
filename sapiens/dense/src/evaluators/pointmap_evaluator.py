# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from sapiens.engine.evaluators import BaseEvaluator
from sapiens.registry import MODELS


@MODELS.register_module()
class PointmapEvaluator(BaseEvaluator):
    def __init__(
        self,
        distance_thresholds: List[float] = [0.05, 0.10, 0.20],
        angle_thresholds: List[float] = [5.0, 11.25, 22.5, 30.0],
    ):
        super().__init__()
        self.distance_thresholds = distance_thresholds
        self.angle_thresholds = angle_thresholds

    def _compute_surface_normals(
        self, point_map: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute surface normals from a point map.

        Args:
            point_map (torch.Tensor): Point map of shape (H, W, 3).
            valid_mask (torch.Tensor): Boolean mask of valid points of shape (H, W).

        Returns:
            torch.Tensor: Surface normals for valid points of shape (N, 3).
        """
        points_np = point_map.cpu().numpy().astype(np.float32)

        grad_x = cv2.Sobel(points_np, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(points_np, cv2.CV_32F, 0, 1, ksize=5)

        normals = np.cross(grad_x, grad_y)
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (norms + 1e-6)

        normals_tensor = torch.from_numpy(normals).to(point_map.device)
        valid_normals = normals_tensor[valid_mask]

        return valid_normals

    @torch.no_grad()
    def process(self, predictions: tuple, data_samples: dict, accelerator=None):
        """
        Process a single batch of predictions and ground truth data.

        Args:
            predictions (tuple): A tuple containing the predicted pointmap and scale.
            data_samples (List[Dict]): A list of dictionaries, each containing ground truth data.
        """
        assert accelerator is not None, "evaluation process expects an accelerator"
        pred_pointmaps, _ = predictions  ## gt pointmaps are canonicalized
        gt_masks = data_samples["mask"]  # B x 1 x H x W
        gt_pointmaps = data_samples["gt_pointmap"]  # B x 3 x H x W

        if pred_pointmaps.shape[2:] != gt_pointmaps.shape[2:]:
            pred_pointmaps = F.interpolate(
                input=pred_pointmaps,
                size=gt_pointmaps.shape[2:],
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

        B = gt_pointmaps.shape[0]
        D = len(self.distance_thresholds)
        A = len(self.angle_thresholds)
        per_sample_vecs = []  # (B_local, K)

        for i in range(B):
            pred_pm = pred_pointmaps[i].permute(1, 2, 0)  # (H, W, 3)
            gt_pm = gt_pointmaps[i].permute(1, 2, 0)  # (H, W, 3)
            valid = gt_masks[i][0] > 0  # (H, W) bool

            # Keep shapes consistent even if there are no valid points
            if valid.any():
                gt_pts = gt_pm[valid]  # (N, 3)
                pred_pts = pred_pm[valid]  # (N, 3)
                diff = pred_pts - gt_pts  # (N, 3)

                # distances & axis errors
                distances = torch.norm(diff, dim=-1)  # (N,)
                axis_err = torch.abs(diff)  # (N, 3)

                # normals (computed from *full* maps, then masked)
                gt_normals = self._compute_surface_normals(gt_pm, valid)  # (N, 3)
                pred_normals = self._compute_surface_normals(pred_pm, valid)  # (N, 3)
                dot = (gt_normals * pred_normals).sum(dim=1).clamp(-1.0, 1.0)
                angles = torch.acos(dot) * (180.0 / np.pi)  # (N,)

                num_points = float(distances.shape[0])

                # base sums
                l2_sum = distances.sum()
                x_abs_sum = axis_err[:, 0].sum()
                y_abs_sum = axis_err[:, 1].sum()
                z_abs_sum = axis_err[:, 2].sum()
                squared_dist_sum = (distances**2).sum()
                angle_sum = angles.sum()
                squared_angle_sum = (angles**2).sum()

                # threshold counts
                dist_counts = [(distances < t).sum() for t in self.distance_thresholds]
                ang_counts = [(angles < t).sum() for t in self.angle_thresholds]

            else:
                # all zeros when no valid points
                l2_sum = x_abs_sum = y_abs_sum = z_abs_sum = 0.0
                squared_dist_sum = angle_sum = squared_angle_sum = 0.0
                num_points = 0.0
                dist_counts = [0.0] * D
                ang_counts = [0.0] * A

            # assemble fixed-length vector on the same device
            vec_list = [
                l2_sum,
                x_abs_sum,
                y_abs_sum,
                z_abs_sum,
                squared_dist_sum,
                angle_sum,
                squared_angle_sum,
                num_points,
                *dist_counts,
                *ang_counts,
            ]
            vec = torch.tensor(
                [float(v) for v in vec_list],
                device=pred_pointmaps.device,
                dtype=torch.float64,
            )  # stable accumulation
            per_sample_vecs.append(vec)

        # (B_local, K)
        pack = torch.stack(per_sample_vecs, dim=0)

        # Global per-step totals via Accelerate (dedups final step automatically)
        gpack = accelerator.gather_for_metrics(pack)  # (B_global_this_step, K)
        step_totals = gpack.sum(dim=0)  # (K,)

        if accelerator.is_main_process:
            self.results.append(step_totals)  # store one vector per step on rank-0
        return

    def evaluate(self, logger=None, accelerator=None) -> Dict[str, float]:
        """
        Compute and log the final metrics after processing all batches.

        Returns:
            Dict[str, float]: A dictionary of the final computed metrics.
        """
        assert accelerator is not None, "evaluation aggregation expects an accelerator"

        if not accelerator.is_main_process:
            self.reset()
            return {}

        if not self.results:
            if logger is not None:
                logger.info("No results to evaluate.")
            return {}

        totals_vec = torch.stack(self.results, dim=0).sum(dim=0)  # (K,)
        D = len(self.distance_thresholds)
        A = len(self.angle_thresholds)

        # unpack
        idx = 0
        l2_sum = totals_vec[idx]
        idx += 1
        x_abs_sum = totals_vec[idx]
        idx += 1
        y_abs_sum = totals_vec[idx]
        idx += 1
        z_abs_sum = totals_vec[idx]
        idx += 1
        squared_dist_sum = totals_vec[idx]
        idx += 1
        angle_sum = totals_vec[idx]
        idx += 1
        squared_angle_sum = totals_vec[idx]
        idx += 1
        num_points = totals_vec[idx]
        idx += 1

        dist_counts = totals_vec[idx : idx + D]
        idx += D
        ang_counts = totals_vec[idx : idx + A]
        idx += A

        total_points = float(num_points.item())
        if total_points <= 0:
            if logger is not None:
                logger.info("No valid points found to evaluate.")
            self.reset()
            return {}

        # metrics
        metrics: Dict[str, float] = {}
        metrics["l2_mean"] = (l2_sum / total_points).item()
        metrics["x_mae"] = (x_abs_sum / total_points).item()
        metrics["y_mae"] = (y_abs_sum / total_points).item()
        metrics["z_mae"] = (z_abs_sum / total_points).item()
        metrics["rmse"] = torch.sqrt(squared_dist_sum / total_points).item()
        metrics["normal_mae"] = (angle_sum / total_points).item()
        metrics["normal_rmse"] = torch.sqrt(squared_angle_sum / total_points).item()

        for i, t in enumerate(self.distance_thresholds):
            out_key = f"within_{int(t * 100):02d}_cm"
            metrics[out_key] = (dist_counts[i] / total_points).item()

        for j, t in enumerate(self.angle_thresholds):
            suf = str(t).replace(".", "_")
            out_key = f"within_{suf}_deg"
            metrics[out_key] = (ang_counts[j] / total_points).item()

        # pretty print
        table = PrettyTable()
        table.field_names = list(metrics.keys())
        table.add_row([f"{float(val):.5f}" for val in metrics.values()])
        if logger is not None:
            logger.info("\n" + table.get_string())

        self.reset()
        return metrics
