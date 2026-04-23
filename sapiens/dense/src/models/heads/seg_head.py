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
class SegHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
        deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        num_classes: int = 29,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels

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

        self.num_classes = num_classes
        self.conv_seg = nn.Conv2d(in_channels, self.num_classes, kernel_size=1)

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
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        out = self.conv_seg(x)
        return out

    def loss(
        self,
        seg_logits: Tensor,
        data_samples: dict,
    ) -> dict:
        seg_labels = data_samples["gt_seg"]  ## B x 1 x H x W; torch.int64

        if seg_logits.shape[2:] != seg_labels.shape[2:]:
            seg_logits = F.interpolate(
                input=seg_logits,
                size=seg_labels.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        seg_labels = seg_labels.squeeze(dim=1)  ## remove the fake dimension, B x H x W

        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_labels,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_labels,
                )
        loss["acc_seg"] = self.accuracy(seg_logits, seg_labels)

        return loss, seg_logits

    def accuracy(self, pred, target, topk=1, thresh=None, ignore_index=255):
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.size(0) == 0:
            accu = [pred.new_tensor(0.0) for i in range(len(topk))]
            return accu[0] if return_single else accu
        assert pred.ndim == target.ndim + 1
        assert pred.size(0) == target.size(0)
        assert maxk <= pred.size(1), (
            f"maxk {maxk} exceeds pred dimension {pred.size(1)}"
        )
        pred_value, pred_label = pred.topk(maxk, dim=1)
        # transpose to shape (maxk, N, ...)
        pred_label = pred_label.transpose(0, 1)
        correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
        if thresh is not None:
            # Only prediction values larger than thresh are counted as correct
            correct = correct & (pred_value > thresh).t()

        if ignore_index is not None:
            correct = correct[:, target != ignore_index]
        res = []
        eps = torch.finfo(torch.float32).eps
        for k in topk:
            # Avoid causing ZeroDivisionError when all pixels
            # of an image are ignored
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
            if ignore_index is not None:
                total_num = target[target != ignore_index].numel() + eps
            else:
                total_num = target.numel() + eps
            res.append(correct_k.mul_(100.0 / total_num))
        return res[0] if return_single else res
