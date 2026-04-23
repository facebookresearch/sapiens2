# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import DATASETS

from .pointmap_base_dataset import PointmapBaseDataset


##-----------------------------------------------------------------------
@DATASETS.register_module()
class PointmapRenderPeopleDataset(PointmapBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
