# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Tuple, Union

from sapiens.registry import DATASETS

from .base_dataset import BaseDataset


@DATASETS.register_module()
class CombinedDataset(BaseDataset):
    def __init__(
        self, datasets: list, pipeline: List[Union[dict, Callable]] = [], **kwargs
    ):
        self.datasets = []
        for cfg in datasets:
            dataset = DATASETS.build(cfg)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        self._len = sum(self._lens)

        super(CombinedDataset, self).__init__(pipeline=pipeline, **kwargs)
        assert len(self.datasets) > 0
        return

    def __len__(self):
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
        if index >= len(self) or index < -len(self):
            raise ValueError(f"index {index} out of bounds for length {len(self)}.")

        if index < 0:
            index = index + len(self)

        subset_index = 0
        while index >= self._lens[subset_index]:
            index -= self._lens[subset_index]
            subset_index += 1
        return subset_index, index

    def prepare_data(self, idx: int) -> Any:
        data_info = self.get_data_info(idx)

        if data_info is None:
            return None

        for transform in self.pipeline.transforms:
            data_info = transform(data_info)

        return data_info

    def get_data_info(self, idx: int) -> dict:
        subset_idx, sample_idx = self._get_subset_index(idx)
        data_info = self.datasets[subset_idx][sample_idx]
        return data_info
