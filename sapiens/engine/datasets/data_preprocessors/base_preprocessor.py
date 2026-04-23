# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, List, Mapping, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS

CastData = Union[torch.Tensor, Mapping, Sequence, str, bytes, None]


# -------------------------------------------------------------------------------
@MODELS.register_module()
class BasePreprocessor(nn.Module):
    def __init__(self, non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")

    def is_seq_of(
        self,
        seq: Any,
        expected_type: Union[Type, tuple],
        seq_type: Optional[Type] = None,
    ) -> bool:
        if seq_type is None:
            exp_seq_type = Sequence
        else:
            assert isinstance(seq_type, type)
            exp_seq_type = seq_type
        if not isinstance(seq, exp_seq_type):
            return False
        for item in seq:
            if not isinstance(item, expected_type):
                return False
        return True

    def stack_batch(
        self,
        tensor_list: List[torch.Tensor],
        pad_size_divisor: int = 1,
        pad_value: Union[int, float] = 0,
    ) -> torch.Tensor:
        if not tensor_list:
            raise ValueError("tensor_list cannot be empty")
        if len({t.ndim for t in tensor_list}) != 1:
            raise ValueError("All tensors must have same number of dimensions")

        ndim = tensor_list[0].ndim
        shapes = torch.tensor([list(t.shape) for t in tensor_list])
        max_dims = (
            torch.ceil(torch.max(shapes, dim=0)[0] / pad_size_divisor)
            * pad_size_divisor
        )

        # Don't pad channel dimension
        pad_amounts = max_dims - shapes
        pad_amounts[:, 0] = 0

        if pad_amounts.sum() == 0:
            return torch.stack(tensor_list)

        # Create padding tuples and pad tensors
        padded = []
        for i, tensor in enumerate(tensor_list):
            pad_tuple = []
            for j in range(ndim - 1, -1, -1):  # Reverse order for F.pad
                pad_tuple.extend([0, int(pad_amounts[i, j])])
            padded.append(F.pad(tensor, pad_tuple, value=pad_value))

        return torch.stack(padded)

    def cast_data(self, data: CastData, device: torch.device) -> CastData:
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key], device) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample, device) for sample in data)
        elif isinstance(data, (torch.Tensor)):
            return data.to(device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        raise NotImplementedError(
            "The forward method must be implemented by a subclass."
        )
