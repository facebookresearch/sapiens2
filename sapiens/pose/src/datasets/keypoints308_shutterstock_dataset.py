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

with open(os.devnull, "w") as f, redirect_stderr(f):
    try:
        from care.data.io import typed
    except Exception:
        pass


@DATASETS.register_module()
class Keypoints308ShutterstockDataset(PoseBaseDataset):
    METAINFO: dict = dict(
        from_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "_base_",
            "keypoints308.py",
        )
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.remove_teeth = self.metainfo["remove_teeth"]
        if self.remove_teeth:
            self.teeth_ids = self.metainfo["teeth_keypoint_ids"]

        return

    def load_data_list(self) -> List[dict]:
        """Load data list from 344 body points."""
        self._register_airstore_handler()
        with open(self.ann_file, "rb") as f:
            raw = f.read()
        raw_data = json.loads(raw)  # samples=5,267,269

        data_list = []
        for i, sample in enumerate(raw_data):
            if "sample_id" not in sample:
                sample["sample_id"] = sample["airstore_id"]

            dp = {
                "airstore_id": sample["sample_id"],
                "img_id": i,
            }
            if sample.get("box-default") is not None:
                dp["box"] = sample["box-default"]
            data_list.append(dp)
        return data_list

    def _register_airstore_handler(self) -> None:
        from typedio.file_system.airstore_client import register_airstore_in_fsspec

        register_airstore_in_fsspec()
        self.path_template = (
            "airstoreds://rlr_detection_services_ml_datasets_no_user_data"
        )
        self.airstore = True

    def _read_from_airstore(self, asset: str, sid: str) -> io.BytesIO:
        with typed.open(self.path_template + f"/{asset}?sampleId={sid}").open() as f:
            data = io.BytesIO(f.read())
        return data

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        try:
            img = Image.open(
                self._read_from_airstore("image", data_info["airstore_id"])
            )  ## pillow image
            keypoints_np = np.load(
                self._read_from_airstore("keypoint", data_info["airstore_id"])
            )  # shape 3 x 344
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

        img = np.array(img)  ## RGB image
        img = img[
            :, :, ::-1
        ]  # Convert RGB to BGR, the model preprocessor will convert this to rgb again

        img_w, img_h = img.shape[1], img.shape[0]

        # process keypoints
        keypoints = keypoints_np[:2].T.reshape(1, -1, 2)  # shape 1 x 344 x 2
        keypoints_visible = np.where(keypoints_np[2].T > 0, 1, 0).reshape(
            1, -1
        )  # shape 1 x 344

        # Identify keypoints that are out of bounds for x (width) and y (height)
        out_of_bounds_w = np.logical_or(
            keypoints[0, :, 0] <= 0, keypoints[0, :, 0] >= img_w
        )
        out_of_bounds_h = np.logical_or(
            keypoints[0, :, 1] <= 0, keypoints[0, :, 1] >= img_h
        )

        # Update keypoints_visible based on the out-of-bounds keypoints
        keypoints_visible[0, out_of_bounds_w | out_of_bounds_h] = 0  # shape 1 x 344
        keypoints[keypoints_visible == 0] = 0

        ## remove teeth keypoints
        if self.remove_teeth:
            # Use numpy's boolean indexing to remove keypoints
            mask = np.ones(keypoints.shape[1], dtype=bool)
            mask[self.teeth_ids] = False
            keypoints = keypoints[:, mask, :]
            keypoints_visible = keypoints_visible[:, mask]

        # Default bounding box to the full image size
        bbox = np.array([0, 0, img_w, img_h], dtype=np.float32).reshape(1, 4)

        if np.any(keypoints_visible):  # If any keypoints are visible
            visible_keypoints = keypoints[0][
                keypoints_visible[0] == 1
            ]  # Filter out the invisible keypoints

            # Get the bounding box encompassing the keypoints
            x_min, y_min = np.clip(
                np.min(visible_keypoints, axis=0), [0, 0], [img_w, img_h]
            )
            x_max, y_max = np.clip(
                np.max(visible_keypoints, axis=0), [0, 0], [img_w, img_h]
            )

            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32).reshape(
                1, 4
            )

        num_keypoints = np.count_nonzero(keypoints_visible)

        ## atleast 8 vis keypoints
        if num_keypoints < self.metainfo["min_visible_keypoints"]:
            random_idx = np.random.randint(0, len(self.data_list))
            return self.get_data_info(random_idx)

        ## check body keypoints additionally
        num_body_keypoints = np.count_nonzero(keypoints_visible[0, :21])
        if num_body_keypoints < 6:
            return None

        ## ignore greyscale images for training
        B, G, R = cv2.split(img)
        if np.array_equal(B, G) and np.array_equal(B, R):
            random_idx = np.random.randint(0, len(self.data_list))
            return self.get_data_info(random_idx)

        data_info = {
            "img": img,
            "img_id": data_info["img_id"],
            "img_path": "",
            "airstore_id": data_info["airstore_id"],
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
            "num_keypoints": num_keypoints,
            "keypoints": keypoints,
            "keypoints_visible": keypoints_visible,
            "iscrowd": 0,
            "segmentation": None,
            "id": idx,
            "category_id": 1,
        }

        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            "upper_body_ids",
            "lower_body_ids",
            "flip_pairs",
            "dataset_keypoint_weights",
            "flip_indices",
            "skeleton_links",
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                "exists in the `data_info`."
            )

            data_info[key] = deepcopy(self.metainfo[key])

        return data_info
