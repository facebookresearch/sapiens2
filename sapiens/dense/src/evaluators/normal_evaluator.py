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
class NormalEvaluator(BaseEvaluator):
    def __init__(
        self,
        angle_thresholds: list[float] = [5.0, 11.25, 22.5, 30.0],
        hist_bin_size_deg: float = 0.5,
        hist_max_deg: float = 180.0,
    ):
        super().__init__()
        self.angle_thresholds = angle_thresholds
        self.hist_bin_size_deg = float(hist_bin_size_deg)
        self.hist_max_deg = float(hist_max_deg)

        # number of histogram bins, edges computed on demand
        self._num_bins = int(
            torch.ceil(torch.tensor(self.hist_max_deg / self.hist_bin_size_deg)).item()
        )

    @torch.no_grad()
    def process(self, predictions: torch.Tensor, data_samples: dict, accelerator=None):
        """
        Process a single batch of predictions and ground truth data.

        Args:
            predictions (tuple): A tuple containing the predicted pointmap and scale.
            data_samples (List[Dict]): A list of dictionaries, each containing ground truth data.
        """
        assert accelerator is not None, "evaluation process expects an accelerator"
        pred_normals = predictions  ## pred normals, B x 3 x H_low x W_low
        gt_masks = data_samples["mask"]  # B x 1 x H x W
        gt_normals = data_samples["gt_normal"]  # B x 3 x H x W

        if pred_normals.shape[2:] != gt_normals.shape[2:]:
            pred_normals = F.interpolate(
                input=pred_normals,
                size=gt_normals.shape[2:],
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

        ## normalize
        eps = 1e-6
        pred_normals = pred_normals / pred_normals.norm(dim=1, keepdim=True).clamp_min(
            eps
        )
        gt_normals = gt_normals / gt_normals.norm(dim=1, keepdim=True).clamp_min(eps)

        B = gt_normals.shape[0]
        HN = self._num_bins

        # packed vector layout:
        # [ sum_angle, sum_angle2, N, counts(<th_0..A-1), hist(0..HN-1) ]
        per_sample_vecs = []

        for i in range(B):
            valid = gt_masks[i, 0] > 0
            n_valid = int(valid.sum().item())

            assert n_valid > 0, "no valid pixels found"

            gt = gt_normals[i].permute(1, 2, 0)[valid]  # (N,3)
            pr = pred_normals[i].permute(1, 2, 0)[valid]  # (N,3)

            dot = (gt * pr).sum(dim=1)  # (N,)
            dot = dot.clamp(-1.0, 1.0)
            angle = torch.acos(dot) * (180.0 / torch.pi)  # (N,)

            ## sums
            sum_angle = angle.sum().to(torch.float64).unsqueeze(0)  # shape (1,)
            sum_angle2 = (
                (angle * angle).sum().to(torch.float64).unsqueeze(0)
            )  # shape (1,)
            N_tensor = torch.tensor(
                [float(n_valid)], dtype=torch.float64, device=pred_normals.device
            )  # (1,)

            ## thresholds
            th_counts = torch.stack(
                [(angle < t).sum().to(torch.float64) for t in self.angle_thresholds],
                dim=0,
            )

            ## histogram
            idx = torch.floor(angle / self.hist_bin_size_deg).long().clamp_(0, HN - 1)
            hist = torch.bincount(idx, minlength=HN).to(torch.float64)

            vec = torch.cat(
                [sum_angle, sum_angle2, N_tensor, th_counts, hist], dim=0
            )  # (K,)
            per_sample_vecs.append(vec)

        # (B_local, K)
        pack = torch.stack(per_sample_vecs, dim=0)
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
        A = len(self.angle_thresholds)
        HN = self._num_bins

        idx = 0
        sum_angle = totals_vec[idx]
        idx += 1
        sum_angle2 = totals_vec[idx]
        idx += 1
        n_total = totals_vec[idx]
        idx += 1
        ang_counts = totals_vec[idx : idx + A]
        idx += A
        hist_counts = totals_vec[idx : idx + HN]
        idx += HN

        # Core metrics
        mae = (sum_angle / n_total).item()
        rmse = torch.sqrt(sum_angle2 / n_total).item()
        within = (ang_counts / n_total * 100.0).tolist()

        # Global median from histogram (bin center)
        cdf = torch.cumsum(hist_counts, dim=0)
        mid = 0.5 * n_total
        bin_idx = torch.searchsorted(cdf, mid).clamp(max=HN - 1).item()
        bin_lo = bin_idx * self.hist_bin_size_deg
        bin_hi = (bin_idx + 1) * self.hist_bin_size_deg
        median = 0.5 * (bin_lo + bin_hi)

        # Assemble metrics dict
        metrics: Dict[str, float] = {
            "normal_mae": mae,
            "normal_median_deg": float(median),
            "normal_rmse": rmse,
        }
        for j, t in enumerate(self.angle_thresholds):
            suf = str(t).replace(".", "_")
            metrics[f"within_{suf}_deg"] = float(within[j])

        # Pretty print
        table = PrettyTable()
        table.field_names = list(metrics.keys())
        table.add_row([f"{float(v):.5f}" for v in metrics.values()])
        if logger is not None:
            logger.info("\n" + table.get_string())

        self.reset()
        return metrics
