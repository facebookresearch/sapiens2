# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction="none")
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(
        self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), f"mask and target have mismatched shapes {mask.shape} v.s.{target.shape}"

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (
                target_weights.ndim in (2, 4)
                and target_weights.shape == target.shape[: target_weights.ndim]
            ), (
                "target_weights and target have mismatched shapes "
                f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    def __init__(
        self, use_target_weight: bool = False, topk: int = 8, loss_weight: float = 1.0
    ):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction="none")
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        ohkm_loss = 0.0
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor, target_weights: Tensor) -> Tensor:
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(
                f"topk ({self.topk}) should not be "
                f"larger than num_keypoints ({num_keypoints})."
            )

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(
                    self.criterion(
                        output[:, idx] * target_weight, target[:, idx] * target_weight
                    )
                )
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight
