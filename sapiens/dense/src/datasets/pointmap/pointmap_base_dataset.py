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
class PointmapBaseDataset(BaseDataset):
    def __init__(self, num_samples=None, **kwargs) -> None:
        self.num_samples = num_samples
        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        data_list = []

        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.mask_dir = os.path.join(self.data_root, "mask")
        self.depth_dir = os.path.join(self.data_root, "depth")
        self.K_dir = os.path.join(self.data_root, "camera_intrinsics")
        self.M_dir = os.path.join(
            self.data_root, "camera_extrinsics"
        )  ## cv camera extrinsics

        print(f"\033[92mLoading {self.__class__.__name__}!\033[0m")

        # Create a set of common file names from all three directories
        rgb_files = {x for x in os.listdir(self.rgb_dir) if x.endswith(".png")}
        mask_files = {x for x in os.listdir(self.mask_dir) if x.endswith(".png")}
        depth_files = {
            x.replace(".npy", ".png")
            for x in os.listdir(self.depth_dir)
            if x.endswith(".npy")
        }
        K_files = {
            x.replace(".txt", ".png")
            for x in os.listdir(self.K_dir)
            if x.endswith(".txt")
        }
        M_files = {
            x.replace(".txt", ".png")
            for x in os.listdir(self.M_dir)
            if x.endswith(".txt")
        }

        # Find the intersection of file names between images, masks, and normals
        common_names = rgb_files & mask_files & depth_files & K_files & M_files

        # Create data list using the common file names
        data_list = [
            {
                "rgb_path": os.path.join(self.rgb_dir, name),
                "mask_path": os.path.join(self.mask_dir, name),
                "depth_path": os.path.join(
                    self.depth_dir, name.replace(".png", ".npy")
                ),
                "K_path": os.path.join(self.K_dir, name.replace(".png", ".txt")),
                "M_path": os.path.join(self.M_dir, name.replace(".png", ".txt")),
            }
            for name in sorted(common_names)
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
                image = cv2.imread(data_info["rgb_path"])  ## bgr image is default
                mask = cv2.imread(data_info["mask_path"])
                depth = np.load(data_info["depth_path"])  ## H x W, ## is not in 0 to 1
                K = np.loadtxt(data_info["K_path"])  ## intrinsics, 3 x 3
                M = np.loadtxt(data_info["M_path"])  ## extrinsics, 4 x 4

        except Exception as e:
            return None

        mask = mask[:, :, 0]  ##

        if image is None or mask is None or depth is None:
            return None

        ## remove any nan depth from valid pixels
        nan_depth = np.isnan(depth)
        if np.any(nan_depth):
            mask[nan_depth] = 0

        if mask.sum() < 10:
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find the bounding box's bounds
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        data_info = {
            "img": image,
            "id": idx,
            "orig_img_height": image.shape[0],
            "orig_img_width": image.shape[1],
            "img_id": os.path.basename(data_info["rgb_path"]),
            "img_path": data_info["rgb_path"],
            "gt_depth": depth,
            "K": K,
            "M": M,
            "mask": mask,
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
        }

        return data_info
