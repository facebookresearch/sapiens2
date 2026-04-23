# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS

from .utils import weight_reduce_loss


def cross_entropy(
    pred,
    label,
    weight=None,
    class_weight=None,
    reduction="mean",
    avg_factor=None,
    ignore_index=255,
):
    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index
    )

    if avg_factor is None and reduction == "mean":
        if class_weight is None:
            avg_factor = label.numel()

        else:
            # label_weights = torch.tensor(
            #     [class_weight[cls] for cls in label], device=class_weight.device
            # )
            label_weights = class_weight[label]
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        loss_name="loss_ce",
        ignore_index=255,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = cross_entropy
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index,
            **kwargs,
        )
        return loss_cls

    @property
    def loss_name(self):
        return self._loss_name


# -------------------------------------------------------------------------------
@MODELS.register_module()
class DiceLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,  # set True for binary single-logit heads
        activate: bool = True,  # apply sigmoid/softmax inside the loss
        reduction: str = "mean",  # "none" | "mean" | "sum"
        naive_dice: bool = False,  # if True: (2TP)/(P+G); else: (2TP)/(||P||^2+||G||^2)
        loss_weight: float = 1.0,
        include_background: bool = False,
        ignore_index: Union[int, None] = 255,
        eps: float = 1e-3,
        loss_name: str = "loss_dice",
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.eps = eps
        self._loss_name = loss_name

    def _to_one_hot_and_mask(self, target: torch.Tensor, num_classes: int):
        assert target.dtype == torch.long, (
            f"target must be torch.long, got {target.dtype}"
        )
        B, H, W = target.shape
        if self.ignore_index is None:
            valid_mask = torch.ones(
                (B, 1, H, W), dtype=torch.bool, device=target.device
            )
            t = target
        else:
            valid_mask = (target != self.ignore_index).unsqueeze(1)
            t = torch.where(
                target == self.ignore_index, torch.zeros_like(target), target
            )

        if num_classes == 1:
            # keep 0/1 labels; do NOT clamp to [0,0]
            one_hot = (t == 1).float().unsqueeze(1)  # (B,1,H,W)
        else:
            # now it's safe to clamp to [0, C-1]
            t = t.clamp(min=0, max=max(0, num_classes - 1))
            one_hot = F.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()
        return one_hot, valid_mask

    def _dice_per_sample(self, pred, target_oh, valid_mask):
        valid = valid_mask.to(dtype=pred.dtype)
        pred_m = pred * valid
        gt_m = target_oh * valid

        # Exclude background for multi-class if requested
        if (
            (not self.include_background)
            and (gt_m.size(1) > 1)
            and (not self.use_sigmoid)
        ):
            pred_m = pred_m[:, 1:, ...]
            gt_m = gt_m[:, 1:, ...]

        dims = (2, 3)
        inter = (pred_m * gt_m).sum(dims)
        if self.naive_dice:
            p_sum = pred_m.sum(dims)
            g_sum = gt_m.sum(dims)
            dice_c = (2 * inter + self.eps) / (p_sum + g_sum + self.eps)
        else:
            p_sq = (pred_m * pred_m).sum(dims)
            g_sq = (gt_m * gt_m).sum(dims)
            dice_c = (2 * inter + self.eps) / (p_sq + g_sq + self.eps)

        present = (gt_m.sum(dims) > 0).to(pred.dtype)  # classes present in GT
        denom = present.sum(dim=1).clamp(min=1)
        return (dice_c * present).sum(dim=1) / denom

    def forward(
        self,
        pred: torch.Tensor,  # (B, C, H, W) logits
        target: torch.Tensor,  # (B, H, W) long
        weight: Union[torch.Tensor, None] = None,  # optional per-sample weights (B,)
        avg_factor: Union[int, None] = None,
        reduction_override: Union[str, None] = None,
    ):
        assert pred.dim() == 4 and target.dim() == 3, (
            "Shapes must be (B,C,H,W) and (B,H,W)."
        )
        B, C, H, W = pred.shape
        reduction = reduction_override or self.reduction

        # activate if requested
        if self.activate:
            if C == 1 or self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                pred = pred.softmax(dim=1)

        # one-hot + valid mask
        target_oh, valid = self._to_one_hot_and_mask(target, num_classes=C)

        # per-sample macro dice
        dice_ps = self._dice_per_sample(pred, target_oh, valid)  # (B,)
        loss = 1.0 - dice_ps

        if self.ignore_index is None:
            sample_valid = torch.ones(
                (loss.shape[0],), dtype=loss.dtype, device=loss.device
            )
        else:
            sample_valid = valid.view(valid.size(0), -1).any(dim=1).to(dtype=loss.dtype)

        gt_m = (target_oh * valid).sum(dim=(2, 3)) > 0  # (B, C)
        if (not self.include_background) and (C > 1) and (not self.use_sigmoid):
            gt_m = gt_m[:, 1:]
        has_present = gt_m.any(dim=1).to(loss.dtype)  # (B,)
        effective_mask = sample_valid * has_present

        if weight is None:
            eff_weight = effective_mask
        else:
            if weight.shape != loss.shape:
                raise ValueError(
                    f"weight must be shape {loss.shape}, got {weight.shape}"
                )
            eff_weight = effective_mask * weight

        if reduction == "mean":
            mean_denom = (
                avg_factor
                if (avg_factor is not None)
                else eff_weight.sum().clamp(min=1).item()
            )
        else:
            mean_denom = None

        loss = self.loss_weight * weight_reduce_loss(
            loss * eff_weight, weight=None, reduction=reduction, avg_factor=mean_denom
        )

        return loss

    @property
    def loss_name(self):
        return self._loss_name
