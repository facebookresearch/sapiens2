# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# from sapiens.pose.evaluation.functional import pose_pck_accuracy
from sapiens.registry import MODELS
from torch import nn, Tensor

from ...evaluators.keypoints308_evaluator import pose_pck_accuracy


@MODELS.register_module()
class PoseHeatmapHead(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
        deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        loss_decode: dict = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                deconv_kernel_sizes
            ):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    "be integer sequences with the same length. Got "
                    f"mismatched lengths {deconv_out_channels} and "
                    f"{deconv_kernel_sizes}"
                )

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                conv_kernel_sizes
            ):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    "be integer sequences with the same length. Got "
                    f"mismatched lengths {conv_out_channels} and "
                    f"{conv_kernel_sizes}"
                )

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes,
            )
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        self.conv_pose = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.loss_decode = MODELS.build(loss_decode)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                if hasattr(m, "weight"):
                    nn.init.ones_(m.weight)

    def _make_conv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create convolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(
                    f"Unsupported kernel size {kernel_size} for"
                    "deconvlutional layers in "
                    f"{self.__class__.__name__}"
                )
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.conv_pose(x)
        return x

    def loss(
        self,
        pred_heatmaps: Tensor,
        data_samples: dict,
    ) -> dict:
        gt_heatmaps = data_samples["heatmaps"]  # B x K x H x W
        keypoint_weights = data_samples["keypoint_weights"]  # B x 1 x K
        keypoint_weights = keypoint_weights.squeeze(dim=1)  # B x K

        if pred_heatmaps.dtype != gt_heatmaps.dtype:
            pred_heatmaps = pred_heatmaps.to(gt_heatmaps.dtype)

        ##---------------------------------
        losses = dict()
        loss = self.loss_decode(pred_heatmaps, gt_heatmaps, keypoint_weights)
        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = pose_pck_accuracy(
            output=pred_heatmaps.detach().cpu().float().numpy(),
            target=gt_heatmaps.detach().cpu().float().numpy(),
            mask=keypoint_weights.detach().cpu().float().numpy() > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
        losses.update(acc_pose=acc_pose)
        return losses, pred_heatmaps
