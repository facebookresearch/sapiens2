# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import DATASETS

from .normal_base_dataset import NormalBaseDataset


##-----------------------------------------------------------------------
@DATASETS.register_module()
class NormalTHumanDataset(NormalBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
