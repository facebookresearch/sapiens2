# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
from sapiens.engine.models import BaseModel
from sapiens.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class PointmapEstimator(BaseModel):
    def __init__(
        self,
        backbone: dict = None,
        decode_head: dict = None,
        canonical_focal_length: float = 768.0,
        init_cfg: dict = None,
    ):
        BaseModel.__init__(self, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        self.canonical_focal_length = canonical_focal_length

    def loss(
        self, outputs: Union[Tensor, Tuple[Tensor, ...]], data_samples: dict
    ) -> Tuple[dict, Tensor]:
        losses, pointmaps = self.decode_head.loss(outputs, data_samples)
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars["outputs"] = pointmaps
        return parsed_losses, log_vars

    def forward(self, inputs: Tensor) -> Tensor:
        ## backbone forward returns a list of tensors at different depths, 0 is the final layer
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.backbone, inputs, use_reentrant=False
            )[0]
        else:
            x = self.backbone(inputs)[0]

        x = self.decode_head(x)
        return x
