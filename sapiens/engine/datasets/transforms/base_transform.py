# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float],
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


class BaseTransform(metaclass=ABCMeta):
    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pass
