# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import cv2
import numpy as np


def bbox_xyxy2xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_xywh2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x, y, w, h = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x + w * 0.5, y + h * 0.5])
    scale = np.hstack([w, h]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_cs2xyxy(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, xy + wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_cs2xywh(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (
        -0.5 * input_size[0] * math.cos(rot_rad)
        + 0.5 * input_size[1] * math.sin(rot_rad)
        + 0.5 * scale[0]
    )
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (
        -0.5 * input_size[0] * math.sin(rot_rad)
        - 0.5 * input_size[1] * math.cos(rot_rad)
        + 0.5 * scale[1]
    )
    return warp_mat


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat
