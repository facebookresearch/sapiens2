# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import io
import json
import os
from typing import List

import numpy as np
from PIL import Image
from sapiens.engine.datasets import BaseDataset
from sapiens.registry import DATASETS

with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
    try:
        from care.data.io import typed
        from typedio.file_system.airstore_client import register_airstore_in_fsspec

        register_airstore_in_fsspec()
    except Exception:
        pass


##-----------------------------------------------------------------------
@DATASETS.register_module()
class SegBaseDataset(BaseDataset):
    def __init__(
        self,
        ann_file=None,
        num_samples=None,
        classes=None,
        palette=None,
        source_to_target_index_mapping=None,
        **kwargs,
    ) -> None:
        self.ann_file = ann_file
        self.classes = classes
        self.palette = palette
        self.source_to_target_index_mapping = source_to_target_index_mapping
        self.num_samples = num_samples
        self.path_template = (
            "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        )
        super().__init__(**kwargs)

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sampleId={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def load_data_list(self) -> List[dict]:
        data_list = []

        with open(self.ann_file, "rb") as f:
            raw = f.read()
        raw_data = json.loads(raw)

        print("\033[92mLoading {}!\033[0m".format(self.__class__.__name__))
        data_list = []
        for sample in raw_data:
            dp = {
                "airstore_id": sample["sample_id"],
                "session_id": str(sample["session_id"]),
                "camera_id": str(sample["camera_id"]),
                "frame_id": str(sample["frame_number"]),
            }
            if sample.get("box-default") is not None:
                dp["box"] = sample["box-default"]
            data_list.append(dp)

        data_list = sorted(
            data_list, key=lambda y: (y["session_id"], y["camera_id"], y["frame_id"])
        )

        if self.num_samples is not None:
            data_list = data_list[: self.num_samples]

        print(
            "\033[92mDone! {}. Loaded total samples: {}. Test mode: {}\033[0m".format(
                self.__class__.__name__, len(data_list), self.test_mode
            )
        )
        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])
        try:
            img = Image.open(
                self._read_from_airstore("image", data_info["airstore_id"])
            )  ## pillow image
            segmentation = Image.open(
                self._read_from_airstore("segmentation", data_info["airstore_id"])
            )

        except Exception as e:
            print(
                f"Error loading image/seg {data_info['airstore_id']} in {self.__class__.__name__}. Retrying!"
            )

            return None

        # Important: Convert RGB to BGR, the pretrained model preprocessor will convert this to rgb again
        img = np.array(img)  ## rgb image
        img = img[:, :, ::-1]
        segmentation = np.array(segmentation)

        ##------remove the extra classes---
        if self.source_to_target_index_mapping is not None:
            segmentation = np.vectorize(
                lambda x: self.source_to_target_index_mapping.get(x, 255)
            )(segmentation)

        ## get bbox
        mask = (segmentation > 0).astype("uint8")  ## 2D binary mask
        if mask.sum() < 8 and self.test_mode is False:  # too small mask
            return None

        data_info = {
            "img": img,
            "img_id": "",
            "img_path": data_info["airstore_id"],
            "gt_seg": segmentation,
            "id": idx,
            "orig_img_height": img.shape[0],
            "orig_img_width": img.shape[1],
        }

        return data_info
