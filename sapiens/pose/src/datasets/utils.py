# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sapiens.engine import Config


def parse_pose_metainfo(metainfo: dict):
    if "from_file" in metainfo:
        cfg_file = metainfo["from_file"]
        metainfo = Config.fromfile(cfg_file).dataset_info

    # check data integrity
    assert "dataset_name" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo
    # assert 'joint_weights' in metainfo
    # assert 'sigmas' in metainfo

    # parse metainfo
    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["dataset_name"] = metainfo["dataset_name"]

    if "remove_teeth" in metainfo:
        parsed["remove_teeth"] = metainfo["remove_teeth"]

    if "min_visible_keypoints" in metainfo:
        parsed["min_visible_keypoints"] = metainfo["min_visible_keypoints"]

    if "teeth_keypoint_ids" in metainfo:
        parsed["teeth_keypoint_ids"] = metainfo["teeth_keypoint_ids"]

    if "coco_wholebody_to_goliath_mapping" in metainfo:
        parsed["coco_wholebody_to_goliath_mapping"] = metainfo[
            "coco_wholebody_to_goliath_mapping"
        ]

    if "coco_wholebody_to_goliath_keypoint_info" in metainfo:
        parsed["coco_wholebody_to_goliath_keypoint_info"] = metainfo[
            "coco_wholebody_to_goliath_keypoint_info"
        ]

    if "idx_to_original_idx_mapping" in metainfo:
        parsed["idx_to_original_idx_mapping"] = metainfo["idx_to_original_idx_mapping"]

    # parse keypoint information
    parsed["num_keypoints"] = len(metainfo["keypoint_info"])

    for kpt_id, kpt in metainfo["keypoint_info"].items():
        kpt_name = kpt["name"]
        parsed["keypoint_id2name"][kpt_id] = kpt_name
        parsed["keypoint_name2id"][kpt_name] = kpt_id
        parsed["keypoint_colors"].append(kpt.get("color", [255, 128, 0]))

        kpt_type = kpt.get("type", "")
        if kpt_type == "upper":
            parsed["upper_body_ids"].append(kpt_id)
        elif kpt_type == "lower":
            parsed["lower_body_ids"].append(kpt_id)

        swap_kpt = kpt.get("swap", "")
        if swap_kpt == kpt_name or swap_kpt == "":
            parsed["flip_indices"].append(kpt_name)
        else:
            parsed["flip_indices"].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed["flip_pairs"]:
                parsed["flip_pairs"].append(pair)

    # parse skeleton information
    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for _, sk in metainfo["skeleton_info"].items():
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [96, 96, 255]))

    # parse extra information
    if "joint_weights" in metainfo:
        parsed["dataset_keypoint_weights"] = np.array(
            metainfo["joint_weights"], dtype=np.float32
        )
    if "sigmas" in metainfo:
        parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)

    if "stats_info" in metainfo:
        parsed["stats_info"] = {}
        for name, val in metainfo["stats_info"].items():
            parsed["stats_info"][name] = np.array(val, dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed["flip_pairs"] = _map(
        parsed["flip_pairs"], mapping=parsed["keypoint_name2id"]
    )
    parsed["flip_indices"] = _map(
        parsed["flip_indices"], mapping=parsed["keypoint_name2id"]
    )
    parsed["skeleton_links"] = _map(
        parsed["skeleton_links"], mapping=parsed["keypoint_name2id"]
    )

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(
        parsed["skeleton_link_colors"], dtype=np.uint8
    )

    return parsed
