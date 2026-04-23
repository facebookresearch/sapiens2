# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS
from torch import Tensor


class RMSNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # x: (N, C, H, W) → normalize across C per spatial location
        x = x.movedim(1, -1).contiguous()  # (N, H, W, C)
        x = self.norm(x)  # RMSNorm over last dim = C
        return x.movedim(-1, 1).contiguous()  # (N, C, H, W)


@MODELS.register_module()
class AlbedoHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        channels: int = 16,
        upsample_channels: List[int] = [768, 384, 192, 96],
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        loss_decode=dict(type="L1Loss", loss_weight=1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.channels = channels
        self._build_network(upsample_channels, conv_out_channels, conv_kernel_sizes)
        in_channels = (
            conv_out_channels[-1] if conv_out_channels else upsample_channels[-1]
        )
        self.conv_albedo = nn.Conv2d(in_channels, 3, kernel_size=1)

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(
                f"loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}"
            )

        self._init_weights()

    def _build_network(
        self,
        upsample_channels: List[int],
        conv_out_channels: Optional[Sequence[int]],
        conv_kernel_sizes: Optional[Sequence[int]],
    ) -> None:
        in_channels = self.in_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            RMSNorm2d(
                in_channels
            ),  # nn.InstanceNorm2d(in_channels),  # Normalize first
            nn.SiLU(inplace=True),
        )

        # Progressive upsampling blocks
        up_blocks = []
        cur_ch = in_channels
        for out_ch in upsample_channels:
            up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(cur_ch, out_ch * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),  # ↑ spatial ×2
                    RMSNorm2d(out_ch),  # nn.InstanceNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )

            cur_ch = out_ch
        self.upsample_blocks = nn.Sequential(*up_blocks)

        # optional extra conv layers
        conv_layers = []
        if conv_out_channels and conv_kernel_sizes:
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                conv_layers.extend(
                    [
                        nn.Conv2d(cur_ch, out_ch, k, padding=(k - 1) // 2),
                        RMSNorm2d(out_ch),  # nn.InstanceNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    ]
                )
                cur_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_out", nonlinearity="relu"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_in", nonlinearity="linear"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        x_albedo = self.input_conv(x)
        x_albedo = self.upsample_blocks(x_albedo)
        x_albedo = self.conv_layers(x_albedo)
        albedo = self.conv_albedo(x_albedo)
        return albedo

    def loss(
        self,
        outputs: Tuple[Tensor],
        data_samples: dict,
    ) -> dict:
        pred_albedo = outputs
        gt_albedo = data_samples["gt_albedo"]  ## B x 3 x H x W
        gt_mask = data_samples["mask"]  ## B x 1 x H x W

        if pred_albedo.shape[2:] != gt_albedo.shape[2:]:
            pred_albedo = F.interpolate(
                input=pred_albedo,
                size=gt_albedo.shape[2:],
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

        ##---------------------------------
        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            this_loss = loss_decode(
                pred_albedo,
                gt_albedo,
                valid_mask=gt_mask,
            )

            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = this_loss
            else:
                loss[loss_decode.loss_name] += this_loss

        return loss, pred_albedo
