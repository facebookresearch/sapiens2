# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
from sapiens.registry import DATASETS

from .normal_base_dataset import NormalBaseDataset


##-----------------------------------------------------------------------
@DATASETS.register_module()
class NormalRenderPeopleBodyDataset(NormalBaseDataset):
    def ignore_face_pixels(self, mask, img_path):
        seg_name = os.path.basename(img_path).replace(".png", "_seg.npy")
        seg_path = os.path.join(self.seg_data_root, seg_name)

        if not os.path.exists(seg_path):
            return None

        seg = np.load(seg_path)  ## part segmentation. 28 classes

        if (
            mask.shape[0] != 4096
            or mask.shape[1] != 3072
            or seg.shape[0] != 1024
            or seg.shape[1] != 768
        ):
            return None

        ## nearest neighbor upsample seg
        seg = cv2.resize(
            seg, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        ## modify mask such that only body is considered. remove face_neck + hair. label 2 and 3
        mask[seg == 2] = 0
        mask[seg == 3] = 0
        mask[seg == 23] = 0
        mask[seg == 24] = 0
        mask[seg == 25] = 0
        mask[seg == 26] = 0
        mask[seg == 27] = 0

        return mask
