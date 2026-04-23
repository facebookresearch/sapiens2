# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import json
import os
from contextlib import redirect_stderr
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sapiens.registry import DATASETS

from .pose_base_dataset import PoseBaseDataset


@DATASETS.register_module()
class Keypoints308GoliathEvalDataset(PoseBaseDataset):
    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        ann = raw_data_info["raw_ann_info"]
        img = raw_data_info["raw_img_info"]

        img_path = os.path.join(self.data_root, img["file_name"])
        img_w, img_h = img["width"], img["height"]

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann["bbox"]
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(ann["goliath_wholebody_kpts"]).reshape(
            1, -1, 3
        )  ## 1 z 308 x 3
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2] > 0)

        num_keypoints = ann["num_keypoints"]

        data_info = {
            "img_id": ann["image_id"],
            "img_path": img_path,
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
            "num_keypoints": num_keypoints,
            "keypoints": keypoints,
            "keypoints_visible": keypoints_visible,
            "iscrowd": ann["iscrowd"],
            "segmentation": None,
            "id": ann["id"],
            "category_id": ann["category_id"],
            "raw_ann_info": copy.deepcopy(ann),
        }

        return data_info
