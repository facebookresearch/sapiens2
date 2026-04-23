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


@MODELS.register_module()
class PointmapHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        channels: int = 16,
        upsample_channels: List[int] = [768, 384, 192, 96],
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        scale_conv_out_channels: Optional[Sequence[int]] = (1536, 512, 128),
        scale_conv_kernel_sizes: Optional[Sequence[int]] = (1, 1, 1),
        scale_final_layer: Optional[Sequence[int]] = (48 * 128, 512, 64, 1),
        loss_decode=dict(type="L1Loss", loss_weight=1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.channels = channels

        self._build_network(upsample_channels, conv_out_channels, conv_kernel_sizes)
        if scale_conv_out_channels is not None:
            self.scale_conv_layers = self._make_regression_conv_layers(
                in_channels=self.in_channels,
                layer_out_channels=scale_conv_out_channels,
                layer_kernel_sizes=scale_conv_kernel_sizes,
            )
            self.scale_final_layer = self._make_final_layer(scale_final_layer)

        else:
            self.scale_conv_layers = None
            self.scale_final_layer = None

        # final conv layer to predict pointmap
        in_channels = (
            conv_out_channels[-1] if conv_out_channels else upsample_channels[-1]
        )
        self.conv_pointmap = nn.Conv2d(in_channels, 3, kernel_size=1)

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
            nn.InstanceNorm2d(in_channels),  # Normalize first
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
                    nn.InstanceNorm2d(out_ch),
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
                        nn.InstanceNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    ]
                )
                cur_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)

    def _make_final_layer(self, final_layer: Sequence[int]) -> nn.Module:
        """Create final layer by given parameters."""
        layers = [nn.Flatten()]
        in_features = final_layer[0]

        for i in range(1, len(final_layer)):
            layers.append(nn.Linear(in_features, final_layer[i]))
            if i < len(final_layer) - 1:  # No activation after the last layer
                layers.append(nn.SiLU())
            in_features = final_layer[i]

        return nn.Sequential(*layers)

    def _make_regression_conv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            stride = 2  # Set stride to 2 to reduce resolution by half
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))

            in_channels = out_channels

        return nn.Sequential(*layers)

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
        x_pointmap = self.input_conv(x)
        x_pointmap = self.upsample_blocks(x_pointmap)
        x_pointmap = self.conv_layers(x_pointmap)
        pointmap = self.conv_pointmap(x_pointmap)

        if self.scale_conv_layers is not None:
            x_scale = self.scale_conv_layers(x)
            scale = self.scale_final_layer(
                x_scale
            )  ## B x 1. scale = f_c / f_actual. in pixel spac of fx
        else:
            scale = None

        return pointmap, scale

    def loss(
        self,
        outputs: Tuple[Tensor],
        data_samples: dict,
    ) -> dict:
        pred_pointmap, pred_scale = outputs
        gt_pointmap = data_samples["gt_pointmap"]  ## B x 3 x H x W
        gt_mean_depth = data_samples["gt_mean_depth"]  ## B x 1 x 1 x 1

        # gt_K = data_samples["meta"]["K"]  ## B x 3 x 3
        gt_original_K = data_samples["meta"]["original_K"]  ## B x 3 x 3
        gt_scale = data_samples["meta"]["scale"].view(-1, 1)  ## B x 1
        gt_mask = data_samples["mask"]  ## B x 1 x H x W

        if pred_pointmap.shape[2:] != gt_pointmap.shape[2:]:
            print(
                "Warning: this is not recommended in pointmap, you may get artifacts!"
            )
            print(
                f"pred_pointmap size: {pred_pointmap.shape}, gt_pointmap size: {gt_pointmap.shape}"
            )
            pred_pointmap = F.interpolate(
                input=pred_pointmap,
                size=gt_pointmap.shape[2:],
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

        ## B x 1 x H x W
        pred_depth = pred_pointmap[:, 2].unsqueeze(dim=1)  ## B x 1 x H x W
        gt_depth = gt_pointmap[:, 2].unsqueeze(dim=1)  ## B x 1 x H x W

        for loss_decode in losses_decode:
            ## pointmap consistency loss
            if loss_decode.loss_name == "loss_K_consistency":
                this_loss = loss_decode(
                    pred_pointmap,
                    gt_pointmap,
                    valid_mask=gt_mask,
                    intrinsics=gt_original_K,  ## Caution: using original K for consistency loss. since X/Z and Y/Z ratio is the same
                )
            elif loss_decode.loss_name == "loss_silog":
                this_loss = loss_decode(
                    pred_depth,
                    gt_depth,
                    valid_mask=gt_mask,
                )
            elif loss_decode.loss_name == "loss_normal":
                this_loss = loss_decode(
                    pred_pointmap,
                    gt_pointmap,
                    valid_mask=gt_mask,
                    scale=gt_scale,
                )
            elif loss_decode.loss_name == "loss_scale_l1":
                this_loss = loss_decode(pred_scale, gt_scale)
            elif loss_decode.loss_name in [
                "loss_l1",
                "loss_shift_invariant",
                "loss_multiscale_l1_2",
                "loss_multiscale_l1_4",
            ]:
                this_loss = loss_decode(
                    pred_pointmap / gt_mean_depth,
                    gt_pointmap / gt_mean_depth,
                    valid_mask=gt_mask,
                )
                this_loss = torch.clamp(this_loss, max=4.0)

            else:
                raise NotImplementedError(
                    f"loss {loss_decode.loss_name} is not implemented"
                )

            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = this_loss
            else:
                loss[loss_decode.loss_name] += this_loss

        return loss, (pred_pointmap, pred_scale)
