# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
from sapiens.registry import MODELS
from torch import Tensor

from .utils import weight_reduce_loss


def silog_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    eps: float = 1e-4,
    clamp_eps: float = 1e-4,
    reduction: Union[str, None] = "mean",
    avg_factor: Optional[int] = None,
) -> Tensor:
    pred, target = pred.flatten(1), target.flatten(1)

    if valid_mask is None:
        valid_mask = (target > eps).detach().float()

    valid_mask = valid_mask.flatten(1)
    diff_log = torch.log(target.clamp(min=clamp_eps)) - torch.log(
        pred.clamp(min=clamp_eps)
    )

    valid_mask = (valid_mask > 0) & (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1
    ).clamp(min=clamp_eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(
        min=clamp_eps
    )

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class SiLogLoss(nn.Module):
    def __init__(
        self, reduction="mean", loss_weight=1.0, eps=-100, loss_name="loss_silog"
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        valid_mask=None,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert pred.shape == target.shape, (
            "the shapes of pred "
            f"({pred.shape}) and target ({target.shape}) are mismatch"
        )

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if valid_mask is None:
            valid_mask = (target > self.eps).detach().float()
            valid_mask = valid_mask[:, 0, :, :].unsqueeze(1)  ## B x 1 x H x W

        loss = self.loss_weight * silog_loss(
            pred,
            target,
            valid_mask=valid_mask,
            weight=weight,
            eps=self.eps,
            clamp_eps=1e-4,
            reduction=reduction,
            avg_factor=avg_factor,
        )

        return loss

    @property
    def loss_name(self):
        return self._loss_name
