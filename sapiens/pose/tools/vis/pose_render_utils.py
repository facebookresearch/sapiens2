# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np


def visualize_keypoints(
    image: np.ndarray,  # RGB uint8 H,W,3
    keypoints,  # list[(J,2)]
    keypoints_visible,  # list[(J,), {0/1}]
    keypoint_scores,  # list[(J,)]
    *,
    radius: int = 4,
    thickness: int = -1,
    color=(255, 0, 0),
    kpt_thr: float = 0.3,
    skeleton: list | None = None,  # [(i,j)]
    kpt_color: list | tuple | np.ndarray | None = None,
    link_color: list | tuple | np.ndarray | None = None,
    show_kpt_idx: bool = False,
) -> np.ndarray:
    img = image.copy()
    H, W = img.shape[:2]

    # defaults
    if skeleton is None:
        skeleton = []  # points only
    if kpt_color is None:
        kpt_color = color
    if link_color is None:
        link_color = (0, 255, 0)

    # robust color normalization: supports tuple, list-of-tuples, np.ndarray (N,3) or (3,)
    def _as_color_list(c, n):
        # torch -> numpy
        if hasattr(c, "detach"):
            c = c.detach().cpu().numpy()
        # numpy -> array
        if isinstance(c, np.ndarray):
            if c.ndim == 2 and c.shape[1] == 3:  # (N,3) palette
                return [tuple(int(v) for v in row) for row in c.tolist()]
            if c.size == 3:  # single (3,)
                return [tuple(int(v) for v in c.tolist())] * max(1, n)
        # python containers
        if isinstance(c, (list, tuple)):
            if n and len(c) == n and isinstance(c[0], (list, tuple, np.ndarray)):
                out = []
                for cc in c:
                    cc = np.asarray(cc).reshape(-1)
                    assert cc.size == 3, "Each color must be length-3"
                    out.append(tuple(int(v) for v in cc.tolist()))
                return out
            # single triplet
            c_arr = np.asarray(c).reshape(-1)
            if c_arr.size == 3:
                return [tuple(int(v) for v in c_arr.tolist())] * max(1, n)
        # fallback: red
        return [(255, 0, 0)] * max(1, n)

    J = keypoints[0].shape[0] if keypoints else 0
    kpt_colors = _as_color_list(kpt_color, J)
    link_colors = _as_color_list(link_color, len(skeleton))

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    for kpts, vis, score in zip(keypoints, keypoints_visible, keypoint_scores):
        kpts = np.asarray(kpts, float)
        vis = np.asarray(vis).reshape(-1).astype(bool)
        score = np.asarray(score).reshape(-1)

        # links (draw in RGB; NO channel flip)
        for lk, (i, j) in enumerate(skeleton):
            if i >= len(kpts) or j >= len(kpts):
                continue
            if not (vis[i] and vis[j]):
                continue
            if score[i] < kpt_thr or score[j] < kpt_thr:
                continue

            x1, y1 = map(int, np.round(kpts[i]))
            x2, y2 = map(int, np.round(kpts[j]))
            if not (in_bounds(x1, y1) and in_bounds(x2, y2)):
                continue

            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                link_colors[lk % len(link_colors)],
                thickness=max(1, thickness),
                lineType=cv2.LINE_AA,
            )

        # points
        for j_idx, (xy, v, s) in enumerate(zip(kpts, vis, score)):
            if not v or s < kpt_thr:
                continue
            x, y = map(int, np.round(xy))
            if not in_bounds(x, y):
                continue

            c = kpt_colors[min(j_idx, len(kpt_colors) - 1)]
            cv2.circle(img, (x, y), radius, c, thickness=-1, lineType=cv2.LINE_AA)
            if show_kpt_idx:
                cv2.putText(
                    img,
                    str(j_idx),
                    (x + radius, y - radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    c,
                    1,
                    cv2.LINE_AA,
                )
    return img
