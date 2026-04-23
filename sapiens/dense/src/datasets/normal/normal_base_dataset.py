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
class NormalBaseDataset(BaseDataset):
    def __init__(
        self,
        seg_data_root: str = None,
        num_samples: int = None,
        normal_extension: str = ".npy",
        **kwargs,
    ) -> None:
        self.seg_data_root = seg_data_root
        self.num_samples = num_samples
        self.normal_extension = normal_extension
        assert self.normal_extension in [".npy", ".npz"]
        super().__init__(**kwargs)

        return

    def load_data_list(self) -> List[dict]:
        data_list = []

        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.normal_dir = os.path.join(self.data_root, "normal")
        self.mask_dir = os.path.join(self.data_root, "mask")

        print("\033[92mLoading {}!\033[0m".format(self.__class__.__name__))

        # Create a set of common file names from all three directories
        rgb_files = {x for x in os.listdir(self.rgb_dir) if x.endswith(".png")}
        mask_files = {x for x in os.listdir(self.mask_dir) if x.endswith(".png")}
        normal_files = {
            x.replace(self.normal_extension, ".png")
            for x in os.listdir(self.normal_dir)
            if x.endswith(self.normal_extension)
        }

        # Find the intersection of file names between images, masks, and normals
        common_names = sorted(rgb_files & mask_files & normal_files)

        # Create data list using the common file names
        data_list = [
            {
                "rgb_path": os.path.join(self.rgb_dir, name),
                "mask_path": os.path.join(self.mask_dir, name),
                "normal_path": os.path.join(
                    self.normal_dir, name.replace(".png", self.normal_extension)
                ),
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
                normal = (
                    np.load(data_info["normal_path"])
                    if self.normal_extension == ".npy"
                    else np.load(data_info["normal_path"])["normal"]
                )
                mask = cv2.imread(data_info["mask_path"])
        except Exception as e:
            return None

        mask = mask[:, :, 0]  ## H x W

        if self.seg_data_root is not None:
            mask = self.ignore_face_pixels(mask, data_info["rgb_path"])

        if mask is None or mask.sum() < 16:  ## min pixels is 16
            return None

        ## remove any nan normals from valid pixels
        nan_normals = np.isnan(normal).any(axis=2)
        if np.any(nan_normals):
            mask[nan_normals] = 0

        ## check if the normal are normalized
        normal_valid = normal[mask > 0]
        norm_normal_valid = np.linalg.norm(normal_valid, axis=1)
        tolerance = 1e-6
        is_normalized = np.all(norm_normal_valid > 1 - tolerance) & np.all(
            norm_normal_valid < 1 + tolerance
        )

        if not is_normalized:
            norms = np.linalg.norm(normal, axis=2, keepdims=True)
            norms = np.maximum(norms, 1e-6)  # Adding epsilon to avoid division by zero
            normal = normal / norms

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
            "gt_normal": normal,
            "mask": mask,
            "id": idx,
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
        }

        return data_info

    ## filter our face/hair pixels for supervision
    def ignore_face_pixels(self, mask, img_path):
        seg_name = os.path.basename(img_path).replace(".png", "_seg.npy")
        seg_path = os.path.join(self.seg_data_root, seg_name)

        if not os.path.exists(seg_path):
            return None

        try:
            seg = np.load(seg_path)  ## part segmentation. 28 classes
        except Exception as e:
            return None

        if seg.shape[0] != mask.shape[0] or seg.shape[1] != mask.shape[1]:
            return None

        ## modify mask such that only body is considered. remove face_neck + hair. label 2 and 3
        mask[seg == 2] = 0
        mask[seg == 3] = 0
        mask[seg == 23] = 0
        mask[seg == 24] = 0
        mask[seg == 25] = 0
        mask[seg == 26] = 0
        mask[seg == 27] = 0

        return mask
