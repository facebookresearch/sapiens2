# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS


@MODELS.register_module()
class L1Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, loss_name="loss_l1"):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, valid_mask):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        loss = F.l1_loss(pred, target, reduction="none") * valid_mask  ## B x C x H x W
        loss = loss.sum() / valid_mask.sum().clamp(min=1)
        loss = loss * self.loss_weight

        ## convert nan to 0
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class MultiscaleL1Loss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        loss_name="loss_multiscale_l1",
        scale_factor=2,
        interpolate_mode="bilinear",
        align_corners=False,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name + f"_{scale_factor}"
        self.scale_factor = scale_factor
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        assert self.interpolate_mode in ["bilinear", "bicubic"]

    def forward(self, pred, target, valid_mask):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )

        if valid_mask.dtype == torch.bool:
            valid_mask = valid_mask.float()

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        # Upsample pred and target using the specified interpolation mode
        pred_scaled = F.interpolate(
            pred,
            scale_factor=self.scale_factor,
            mode=self.interpolate_mode,
            align_corners=self.align_corners,
        )
        target_scaled = F.interpolate(
            target,
            scale_factor=self.scale_factor,
            mode=self.interpolate_mode,
            align_corners=self.align_corners,
        )

        # Upsample valid_mask using nearest neighbor to preserve binary values
        valid_mask_scaled = F.interpolate(
            valid_mask, scale_factor=self.scale_factor, mode="nearest"
        )

        loss_scaled = (
            F.l1_loss(pred_scaled, target_scaled, reduction="none") * valid_mask_scaled
        )  # B x C x H' x W'

        loss_scaled = loss_scaled.sum() / valid_mask_scaled.sum().clamp(min=1)
        loss_scaled = loss_scaled * self.loss_weight

        # Convert any NaN or Inf values to 0
        loss_scaled = torch.nan_to_num(loss_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return loss_scaled

    @property
    def loss_name(self):
        return self._loss_name
