# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from sapiens.registry import MODELS

from .base_preprocessor import BasePreprocessor


@MODELS.register_module()
class ImagePreprocessor(BasePreprocessor):
    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(non_blocking)
        self._validate_params(mean, std, bgr_to_rgb, rgb_to_bgr)
        self._setup_normalization(mean, std)
        self._channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def _validate_params(self, mean, std, bgr_to_rgb, rgb_to_bgr):
        if bgr_to_rgb and rgb_to_bgr:
            raise ValueError("Cannot set both bgr_to_rgb and rgb_to_bgr to True")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must both be None or both be provided")

    def _setup_normalization(self, mean, std):
        if mean is None:
            self._enable_normalize = False
            return

        if len(mean) not in [1, 3] or len(std) not in [1, 3]:
            raise ValueError("mean and std must have 1 or 3 values")

        self._enable_normalize = True
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)

    def _process_single_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dtype not in [torch.uint8, torch.float16, torch.float32, torch.float64]:
            raise TypeError(f"Unsupported image dtype: {img.dtype}")

        # Handle batched input (NCHW)
        if img.dim() == 4:
            if img.shape[1] != 3:
                raise ValueError(f"Expected 3 channels in dim=1, got {img.shape}")
            img = img.float()
            if self._channel_conversion:
                img = img[:, [2, 1, 0], ...]  # BGR<->RGB
            if self._enable_normalize:
                img = (img - self.mean[None]) / self.std[None]
            return img

        # Handle single image (CHW)
        elif img.dim() == 3:
            if img.shape[0] != 3:
                raise ValueError(f"Expected 3 channels in dim=0, got {img.shape}")
            img = img.float()
            if self._channel_conversion:
                img = img[[2, 1, 0], ...]
            if self._enable_normalize:
                img = (img - self.mean) / self.std
            return img

        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {img.shape}")

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pad_size_divisor <= 1:
            return tensor

        h, w = tensor.shape[-2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h == 0 and pad_w == 0:
            return tensor

        return F.pad(tensor, (0, pad_w, 0, pad_h), "constant", self.pad_value)

    def forward(self, data: dict) -> dict:
        data = self.cast_data(data, device=self.mean.device)
        inputs = data["inputs"]

        if self.is_seq_of(inputs, torch.Tensor):
            # Process list of individual images
            processed_imgs = [self._process_single_image(img) for img in inputs]
            batch_inputs = self.stack_batch(
                processed_imgs, self.pad_size_divisor, self.pad_value
            )
        elif isinstance(inputs, torch.Tensor):
            # Process batched tensor
            if inputs.dim() == 4:
                batch_inputs = self._process_single_image(inputs)
                batch_inputs = self._pad_tensor(batch_inputs)
            elif inputs.dim() == 5:
                # inputs: (B, V, C, H, W)
                B, V, C, H, W = inputs.shape
                flat_inputs = inputs.view(B * V, C, H, W)

                processed = self._process_single_image(flat_inputs)
                processed = self._pad_tensor(processed)

                batch_inputs = processed.view(
                    B, V, C, processed.shape[-2], processed.shape[-1]
                )
            elif inputs.dim() == 3:
                # Single image (C, H, W), unsqueeze to (1, C, H, W)
                img = inputs.unsqueeze(0)
                processed = self._process_single_image(img)
                batch_inputs = self._pad_tensor(processed)
            else:
                raise ValueError(
                    f"Expected 3D, 4D or 5D tensor, got shape {inputs.shape}"
                )
        else:
            raise TypeError(f"Expected tensor or list of tensors, got {type(inputs)}")

        data["inputs"] = batch_inputs
        data.setdefault("data_samples", None)
        return data
