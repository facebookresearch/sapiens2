# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import DATASETS

from .seg_base_dataset import SegBaseDataset
from .seg_utils import DOME_CLASSES_29, INTERNAL_CLASSES_32, INTERNAL_MAPPING_32_to_29

CLASSES = [DOME_CLASSES_29[i]["name"] for i in range(len(DOME_CLASSES_29))]
PALETTE = [DOME_CLASSES_29[i]["color"] for i in range(len(DOME_CLASSES_29))]

SOURCE_TO_TARGET_INDEX_MAPPING = {
    i: INTERNAL_MAPPING_32_to_29[i]["target_class_idx"]
    for i in range(len(INTERNAL_CLASSES_32))
    if INTERNAL_MAPPING_32_to_29[i]["target_class_idx"] is not None
}


##-----------------------------------------------------------------------
@DATASETS.register_module()
class SegInternalClass29Dataset(SegBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            classes=CLASSES,
            palette=PALETTE,
            source_to_target_index_mapping=SOURCE_TO_TARGET_INDEX_MAPPING,
            **kwargs,
        )
        self.is_itw = True
        return

    def get_data_info(self, idx):
        data = super().get_data_info(idx)

        if data is None:
            return None
        data["is_itw"] = self.is_itw
        return data
