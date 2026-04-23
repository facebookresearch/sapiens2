# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import DATASETS

from .albedo_base_dataset import AlbedoBaseDataset


##-----------------------------------------------------------------------
@DATASETS.register_module()
class AlbedoRenderPeopleDataset(AlbedoBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
