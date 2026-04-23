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
class AlbedoChromaticityL1Loss(nn.Module):
    """
    Brightness-invariant color loss in RGB.
    Per-pixel chroma = RGB / (R+G+B); then L1 on chroma.
    Expects pred/target in [0,1] sRGB, shape (B,3,H,W).
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
        loss_name: str = "loss_chroma_l1",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    @staticmethod
    def _chroma(x, eps):
        s = x.sum(dim=1, keepdim=True)  # (B,1,H,W)
        return x / (s + eps)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        pred = pred.clamp(0, 1)

        pc = self._chroma(pred, self.eps)
        gc = self._chroma(target, self.eps)

        m = valid_mask.to(pred.dtype)  # (B,1,H,W), broadcasts over C
        per_pix = (pc - gc).abs()  # (B,3,H,W)
        num = (per_pix * m).sum()
        den = m.sum().clamp(min=1)
        loss = (num / den) * self.loss_weight
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class AlbedoLowFreqL1Loss(nn.Module):
    """
    Compares heavy low-pass versions; leaves edges alone.
    Expects pred, gt in [0,1] sRGB (range doesn't matter for the low-pass itself).
    """

    def __init__(
        self,
        loss_weight=1.0,
        down_sample: int = 32,
        loss_name="loss_low_l1",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.down_sample = down_sample

    def _lowpass_masked(self, x, m):
        # x:(B,3,H,W), m:(B,1,H,W) in {0,1}
        if self.down_sample == 1:
            return x
        xm = x * m
        x_ds = F.interpolate(xm, scale_factor=1.0 / self.down_sample, mode="area")
        m_ds = F.interpolate(m, scale_factor=1.0 / self.down_sample, mode="area")
        x_up = F.interpolate(
            x_ds, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        m_up = F.interpolate(
            m_ds, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return x_up / (m_up + 1e-6)

    def forward(self, pred, target, valid_mask):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        pred_lp = self._lowpass_masked(pred, valid_mask.to(pred.dtype))
        target_lp = self._lowpass_masked(target, valid_mask.to(pred.dtype))

        if pred_lp.dtype != target_lp.dtype:
            target_lp = target_lp.to(pred_lp.dtype)

        loss = F.l1_loss(pred_lp, target_lp, reduction="none") * valid_mask
        loss = loss.sum() / valid_mask.sum().clamp(min=1)
        loss = loss * self.loss_weight

        # Convert nan to 0
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class AlbedoGradL1Loss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        loss_name="loss_grad_l1",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, valid_mask):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        ## pred is B x C x H x W
        ## target is B x C x H x W
        ## valid_mask is B x 1 x H x W
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        if target_dx.dtype != pred_dx.dtype or target_dy.dtype != pred_dy.dtype:
            # Convert to the same dtype as pred
            target_dx = target_dx.to(pred_dx.dtype)
            target_dy = target_dy.to(pred_dy.dtype)

        # Adjust valid mask for gradients
        valid_mask_dx = valid_mask[:, :, :, :-1] * valid_mask[:, :, :, 1:]
        valid_mask_dy = valid_mask[:, :, :-1, :] * valid_mask[:, :, 1:, :]

        # Compute edge-aware loss
        loss_dx = F.mse_loss(pred_dx, target_dx, reduction="none")
        loss_dy = F.mse_loss(pred_dy, target_dy, reduction="none")

        # Apply valid mask before reduction
        loss_dx = (loss_dx * valid_mask_dx).sum() / valid_mask_dx.sum().clamp(min=1)
        loss_dy = (loss_dy * valid_mask_dy).sum() / valid_mask_dy.sum().clamp(min=1)

        # Combine losses
        loss = (loss_dx + loss_dy) * 0.5 * self.loss_weight

        # Convert nan to 0
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

    @property
    def loss_name(self):
        return self._loss_name
