# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import os
import sys
from typing import List

import cv2
import numpy as np
from sapiens.engine.datasets import BaseDataset
from sapiens.registry import DATASETS


@contextlib.contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = sys.stderr.fileno()
    sys.stderr.flush()
    saved_stderr_fd = os.dup(stderr_fd)

    os.dup2(devnull_fd, stderr_fd)
    os.close(devnull_fd)

    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


##-----------------------------------------------------------------------
@DATASETS.register_module()
class AlbedoBaseDataset(BaseDataset):
    def __init__(self, num_samples: int = None, **kwargs) -> None:
        self.num_samples = num_samples
        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        data_list = []

        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.albedo_dir = os.path.join(self.data_root, "albedo")
        self.mask_dir = os.path.join(self.data_root, "mask")

        print("\033[92mLoading {}!\033[0m".format(self.__class__.__name__))

        # Create a set of common file names from all three directories
        rgb_files = {x for x in os.listdir(self.rgb_dir) if x.endswith(".png")}
        mask_files = {x for x in os.listdir(self.mask_dir) if x.endswith(".png")}
        albedo_files = {x for x in os.listdir(self.albedo_dir) if x.endswith(".png")}

        common_names = sorted(rgb_files & mask_files & albedo_files)

        # Create data list using the common file names
        data_list = [
            {
                "rgb_path": os.path.join(self.rgb_dir, name),
                "mask_path": os.path.join(self.mask_dir, name),
                "albedo_path": os.path.join(self.albedo_dir, name),
            }
            for name in common_names
        ]

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
            with suppress_stderr():
                img = cv2.imread(data_info["rgb_path"])  ## bgr image is default
                albedo = cv2.imread(data_info["albedo_path"])
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)  ## RGB image
                mask = cv2.imread(data_info["mask_path"])

        except Exception:
            return None

        mask = mask[:, :, 0]  ## H x W

        if mask is None or mask.sum() < 16:  ## min pixels is 16
            return None

        # Normalize albedo to the range 0-1
        albedo = albedo.astype(float) / 255.0

        # Check if 98% of the pixels are the same color
        albedo_mask = albedo[mask > 0]
        std_per_channel = np.std(albedo_mask, axis=0)
        if np.max(std_per_channel) < 0.02:  # Threshold can be adjusted
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            "img": img,
            "img_id": os.path.basename(data_info["rgb_path"]),
            "img_path": data_info["rgb_path"],
            "gt_albedo": albedo,  ## rgb format
            "mask": mask,
            "id": idx,
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
        }

        return data_info
