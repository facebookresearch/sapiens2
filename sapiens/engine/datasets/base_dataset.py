# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from sapiens.registry import TRANSFORMS
from torch.utils.data import Dataset


class Compose:
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms = []
        for t in transforms or []:
            if isinstance(t, dict):
                t = TRANSFORMS.build(t)
            if not callable(t):
                raise TypeError(f"Transform must be callable, got {type(t)}")
            self.transforms.append(t)

    def __call__(self, data: dict) -> Optional[dict]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transforms})"


# -------------------------------------------------------------------------------
class BaseDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = "",
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        max_refetch: int = 1000,
    ):
        self.data_root = data_root
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.pipeline = Compose(pipeline)
        self.data_list = self.load_data_list()

    def get_data_info(self, idx: int) -> dict:
        data_info = copy.deepcopy(self.data_list[idx])
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx
        return data_info

    def __getitem__(self, idx: int) -> dict:
        if self.test_mode:
            data_info = self.get_data_info(idx)
            if data_info is None:
                warnings.warn(
                    f"Test time pipeline should not get `None` data_sample, index:{idx}, using idx=0 as default"
                )
                return self.__getitem__(idx=0)
            data = self.pipeline(data_info)
            if data is None:
                warnings.warn(
                    f"Test time pipeline outputs `None` for index:{idx}, using idx=0 as default"
                )
                return self.__getitem__(idx=0)

            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f"Cannot find valid data after {self.max_refetch}! ")

    @abstractmethod
    def load_data_list(self) -> List[dict]:
        pass

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self))

    def __len__(self) -> int:
        return len(self.data_list)

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        if data_info is None:
            return None
        return self.pipeline(data_info)
