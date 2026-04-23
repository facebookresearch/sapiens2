# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import json
import os
import random
from typing import Any, Iterator, List

import cv2
import numpy as np
import torch
from iopath.common.file_io import PathManager
from sapiens.registry import DATASETS

from .normal_base_dataset import NormalBaseDataset, suppress_stderr

try:
    from airstore.client.airstore_tabular import AIRStorePathHandler
except:
    pass


##-----------------------------------------------------------------------
@DATASETS.register_module()
class NormalMetaSimDataset(NormalBaseDataset):
    def __init__(self, airstore_template=None, json_path=None, **kwargs) -> None:
        self.airstore_template = airstore_template
        self.json_path = json_path
        self._cached_iterator = None
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        data_list = []

        self.path_manager = PathManager()
        self.path_manager.register_handler(AIRStorePathHandler())

        print("\033[92mLoading {}!\033[0m".format(self.__class__.__name__))

        ## load json file
        with open(self.json_path, "r") as f:
            data_list = json.load(f)

        if self.num_samples is not None:
            data_list = data_list[: self.num_samples]

        print(
            "\033[92mDone! {}. Loaded total samples: {}\033[0m".format(
                self.__class__.__name__, len(data_list)
            )
        )

        return data_list

    def _open_iterator(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        airstore_rank = self.global_rank * num_workers + worker_id
        random_seed = (
            random.randint(0, 100000) + random.randint(0, 100000) * airstore_rank
        )
        return self.path_manager.opent(
            self.airstore_template, seed=random_seed, enable_shuffle=True
        )

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        if self._cached_iterator is None:
            self._cached_iterator = self._open_iterator()

        while True:
            try:
                with suppress_stderr():
                    row = next(self._cached_iterator)
                    img_buf = np.frombuffer(row["image"], dtype=np.uint8)
                    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)  ## this in BGR format

                    mask_buf = np.frombuffer(row["mask"], dtype=np.uint8)
                    mask = cv2.imdecode(
                        mask_buf, cv2.IMREAD_GRAYSCALE
                    )  ## this in BGR format

                    normal = np.load(io.BytesIO(row["normal"]))["normal"]
                    break

            except Exception as e:
                print(f"Error loading data: {e}, {data_info['rgb_path']}")
                return None

        if mask.sum() < 16:
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
            norms = np.maximum(norms, 1e-6)  # Avoid division by zero
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
