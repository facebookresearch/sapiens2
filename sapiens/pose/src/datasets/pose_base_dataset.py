# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from sapiens.engine.datasets import BaseDataset
from sapiens.registry import DATASETS

from .utils import parse_pose_metainfo


@DATASETS.register_module()
class PoseBaseDataset(BaseDataset):
    METAINFO: dict = dict(from_file="configs/_base_/keypoints308.py")

    def __init__(
        self,
        ann_file: str = "",
        num_samples: int = None,
        bbox_file: Optional[str] = None,
        **kwargs,
    ):
        self.bbox_file = bbox_file
        self.ann_file = ann_file
        self.num_samples = num_samples

        self.metainfo = parse_pose_metainfo(self.METAINFO)
        super().__init__(**kwargs)

        if self.num_samples is not None:
            self.data_list = self.data_list[:num_samples]

        print(
            "\033[96mLoaded {} samples for {}, Test mode: {}\033[0m".format(
                self.__len__(), self.__class__.__name__, self.test_mode
            )
        )
        return

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        transformed_data_info = self.pipeline(data_info)

        if transformed_data_info is None:
            return None

        ## pipeline is set to empty when using concatenation of datasets.
        if (
            self.test_mode == False
            and "data_samples" in transformed_data_info
            and "gt_instance_labels" in transformed_data_info["data_samples"]
            and "keypoints_visible"
            in transformed_data_info["data_samples"].gt_instance_labels
        ):
            num_transformed_keypoints = (
                transformed_data_info["data_samples"]
                .gt_instance_labels["keypoints_visible"]
                .sum()
                .item()
            )  ## after cropping

            ## minimum visible keypoints for coco_wholebody is 8
            if self.metainfo["dataset_name"] == "coco_wholebody":
                if num_transformed_keypoints < 8:
                    return None

            ## absolute minimum visible keypoints is 3
            if num_transformed_keypoints < 3:
                return None

        return transformed_data_info

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info["img"] = cv2.imread(data_info["img_path"])

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

    def load_data_list(self) -> List[dict]:
        if self.bbox_file:
            data_list = self._load_detection_results()
        else:
            instance_list, _ = self._load_annotations()
            data_list = self._get_topdown_data_infos(instance_list)

        return data_list

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        from xtcocotools.coco import COCO  # lazy: only needed for COCO-format ann files

        assert os.path.exists(self.ann_file), "Annotation file does not exist"
        self.coco = COCO(self.ann_file)
        self.metainfo["CLASSES"] = self.coco.loadCats(self.coco.getCatIds())

        instance_list = []
        image_list = []

        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs(img_id)[0]
            img.update(
                {
                    "img_id": img_id,
                    "img_path": os.path.join(self.data_root, img["file_name"]),
                }
            )
            image_list.append(img)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            for ann in self.coco.loadAnns(ann_ids):
                instance_info = self.parse_data_info(
                    dict(raw_ann_info=ann, raw_img_info=img)
                )

                # skip invalid instance annotation.
                if not instance_info:
                    continue

                instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        ann = raw_data_info["raw_ann_info"]
        img = raw_data_info["raw_img_info"]

        # filter invalid instance
        if "bbox" not in ann or "keypoints" not in ann:
            return None

        img_w, img_h = img["width"], img["height"]

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann["bbox"]
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if "num_keypoints" in ann:
            num_keypoints = ann["num_keypoints"]
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            "img_id": ann["image_id"],
            "img_path": img["img_path"],
            "bbox": bbox,
            "bbox_score": np.ones(1, dtype=np.float32),
            "num_keypoints": num_keypoints,
            "keypoints": keypoints,
            "keypoints_visible": keypoints_visible,
            "iscrowd": ann.get("iscrowd", 0),
            "segmentation": ann.get("segmentation", None),
            "id": ann["id"],
            "category_id": ann["category_id"],
            "raw_ann_info": copy.deepcopy(ann),
        }

        if "crowdIndex" in img:
            data_info["crowd_index"] = img["crowdIndex"]

        return data_info

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        # crowd annotation
        if "iscrowd" in data_info and data_info["iscrowd"]:
            return False
        # invalid keypoints
        if "num_keypoints" in data_info and data_info["num_keypoints"] == 0:
            return False
        # invalid bbox
        if "bbox" in data_info:
            bbox = data_info["bbox"][0]
            w, h = bbox[2:4] - bbox[:2]
            if w <= 0 or h <= 0:
                return False
        # invalid keypoints
        if "keypoints" in data_info:
            if np.max(data_info["keypoints"]) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        data_list_tp = list(filter(self._is_valid_instance, instance_list))
        return data_list_tp

    def _load_detection_results(self) -> List[dict]:
        raise NotImplementedError
