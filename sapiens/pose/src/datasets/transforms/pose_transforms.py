# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from sapiens.engine.datasets import BaseTransform, to_tensor
from sapiens.registry import TRANSFORMS
from scipy.stats import truncnorm

from ..codecs.udp_heatmap import UDPHeatmap
from .bbox_transforms import bbox_xyxy2cs, get_udp_warp_matrix, get_warp_matrix

try:
    warnings.filterwarnings(
        "ignore",
        message=r"Error fetching version info",
        category=UserWarning,
        module=r"^albumentations\.check_version$",
    )

    import albumentations

except ImportError:
    albumentations = None

Number = Union[int, float]


@TRANSFORMS.register_module()
class PoseGenerateTarget(BaseTransform):
    def __init__(
        self,
        encoder: None,
        multilevel: bool = False,
        use_dataset_keypoint_weights: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_cfg = deepcopy(encoder)
        self.multilevel = multilevel
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights
        encoder_type = self.encoder_cfg.pop("type")
        assert encoder_type == "UDPHeatmap", "Only UDPHeatmap is supported"
        self.encoder = UDPHeatmap(**self.encoder_cfg)

    def transform(self, results: Dict) -> Optional[dict]:
        if results.get("transformed_keypoints", None) is not None:
            keypoints = results["transformed_keypoints"]  ## N x K x 2
        elif results.get("keypoints", None) is not None:
            keypoints = results["keypoints"]
        else:
            raise ValueError(
                "GenerateTarget requires 'transformed_keypoints' or"
                " 'keypoints' in the results."
            )

        keypoints_visible = results["keypoints_visible"]  ## N x K

        auxiliary_encode_kwargs = {
            key: results[key] for key in self.encoder.auxiliary_encode_keys
        }
        encoded = self.encoder.encode(
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
            **auxiliary_encode_kwargs,
        )

        if self.use_dataset_keypoint_weights and "keypoint_weights" in encoded:
            if isinstance(encoded["keypoint_weights"], list):
                for w in encoded["keypoint_weights"]:
                    w *= results["dataset_keypoint_weights"]
            else:
                encoded["keypoint_weights"] *= results["dataset_keypoint_weights"]

        results.update(encoded)

        if results.get("keypoint_weights", None) is not None:
            results["transformed_keypoints_visible"] = results["keypoint_weights"]
        elif results.get("keypoints", None) is not None:
            results["transformed_keypoints_visible"] = results["keypoints_visible"]
        else:
            raise ValueError(
                "GenerateTarget requires 'keypoint_weights' or"
                " 'keypoints_visible' in the results."
            )

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(encoder={str(self.encoder_cfg)}, "
        repr_str += f"use_dataset_keypoint_weights={self.use_dataset_keypoint_weights})"
        return repr_str


@TRANSFORMS.register_module()
class PoseTopdownAffine(BaseTransform):
    def __init__(self, input_size: Tuple[int, int], use_udp: bool = True) -> None:
        super().__init__()

        assert len(input_size) == 2, f"Invalid input_size {input_size}"

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(
            w > h * aspect_ratio,
            np.hstack([w, w / aspect_ratio]),
            np.hstack([h * aspect_ratio, h]),
        )
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results["bbox_scale"] = self._fix_aspect_ratio(
            results["bbox_scale"], aspect_ratio=w / h
        )

        assert results["bbox_center"].shape[0] == 1, (
            "Top-down heatmap only supports single instance. Got invalid "
            f"shape of bbox_center {results['bbox_center'].shape}."
        )

        center = results["bbox_center"][0]
        scale = results["bbox_scale"][0]
        if "bbox_rotation" in results:
            rot = results["bbox_rotation"][0]
        else:
            rot = 0.0

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        # estimate overall scale from the affine matrix
        sx = np.linalg.norm(warp_mat[0, :2])
        sy = np.linalg.norm(warp_mat[1, :2])
        scale_factor = min(sx, sy)

        # choose interpolation: area for down, linear for up
        interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC

        results["img"] = cv2.warpAffine(
            results["img"], warp_mat, warp_size, flags=interp
        )  ## H x W x 3

        if results.get("keypoints", None) is not None:
            transformed_keypoints = results["keypoints"].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results["keypoints"][..., :2], warp_mat
            )

            ## if transformed_keypoints out of bound, set them to zero
            out_of_bounds = (
                (transformed_keypoints[..., 0] < 0)
                | (transformed_keypoints[..., 0] >= w)
                | (transformed_keypoints[..., 1] < 0)
                | (transformed_keypoints[..., 1] >= h)
            )  ## N x K

            transformed_keypoints[out_of_bounds] = 0  # mask out-of-bound keypoints
            results["transformed_keypoints"] = transformed_keypoints

            # # ## set the visibility of out-of-bound keypoints to 0
            results["keypoints_visible"] = results["keypoints_visible"].copy()
            results["keypoints_visible"][out_of_bounds] = 0

        results["input_size"] = (w, h)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_size={self.input_size}, "
        repr_str += f"use_udp={self.use_udp})"
        return repr_str


@TRANSFORMS.register_module()
class PoseGetBBoxCenterScale(BaseTransform):
    def __init__(self, padding: float = 1.25) -> None:
        super().__init__()
        self.padding = padding

    def transform(self, results: Dict) -> Optional[dict]:
        if "bbox_center" in results and "bbox_scale" in results:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn(
                    'Use the existing "bbox_center" and "bbox_scale"'
                    ". The padding will still be applied."
                )
            results["bbox_scale"] *= self.padding

        else:
            bbox = results["bbox"]
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            results["bbox_center"] = center
            results["bbox_scale"] = scale

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f"(padding={self.padding})"
        return repr_str


@TRANSFORMS.register_module()
class PoseRandomFlip(BaseTransform):
    def __init__(
        self,
        prob: Union[float, List[float]] = 0.5,
        direction: str = "horizontal",
    ) -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(
                f"probs must be float or list of float, but \
                              got `{type(prob)}`."
            )
        self.prob = prob
        self.direction = direction

    def flip_bbox(
        self,
        bbox: np.ndarray,
        image_size: Tuple[int, int],
        bbox_format: str = "xyxy",
        direction: str = "horizontal",
    ) -> np.ndarray:
        format_options = {"xywh", "xyxy", "center"}
        assert bbox_format in format_options, (
            f'Invalid bbox format "{bbox_format}". Options are {format_options}'
        )

        bbox_flipped = bbox.copy()
        w, h = image_size

        if direction == "horizontal":
            if bbox_format == "xywh" or bbox_format == "center":
                bbox_flipped[..., 0] = w - bbox[..., 0] - 1
            elif bbox_format == "xyxy":
                bbox_flipped[..., ::2] = w - bbox[..., ::2] - 1
        elif direction == "vertical":
            if bbox_format == "xywh" or bbox_format == "center":
                bbox_flipped[..., 1] = h - bbox[..., 1] - 1
            elif bbox_format == "xyxy":
                bbox_flipped[..., 1::2] = h - bbox[..., 1::2] - 1
        elif direction == "diagonal":
            if bbox_format == "xywh" or bbox_format == "center":
                bbox_flipped[..., :2] = [w, h] - bbox[..., :2] - 1
            elif bbox_format == "xyxy":
                bbox_flipped[...] = [w, h, w, h] - bbox - 1

        return bbox_flipped

    def flip_keypoints(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray],
        image_size: Tuple[int, int],
        flip_indices: List[int],
        direction: str = "horizontal",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert keypoints.shape[:-1] == keypoints_visible.shape, (
            f"Mismatched shapes of keypoints {keypoints.shape} and "
            f"keypoints_visible {keypoints_visible.shape}"
        )

        direction_options = {"horizontal"}
        assert direction in direction_options, (
            f'Invalid flipping direction "{direction}". Options are {direction_options}'
        )

        # swap the symmetric keypoint pairs
        if direction == "horizontal" or direction == "vertical":
            keypoints = keypoints[..., flip_indices, :]
            if keypoints_visible is not None:
                keypoints_visible = keypoints_visible[..., flip_indices]

        # flip the keypoints
        w, h = image_size
        if direction == "horizontal":
            keypoints[..., 0] = w - 1 - keypoints[..., 0]
        elif direction == "vertical":
            keypoints[..., 1] = h - 1 - keypoints[..., 1]
        else:
            keypoints = [w, h] - keypoints - 1

        return keypoints, keypoints_visible

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            results["flip"] = False
            results["flip_direction"] = ""
            return results

        flip_dir = "horizontal"
        results["flip"] = True
        results["flip_direction"] = flip_dir

        h, w = results["img"].shape[:2]
        results["img"] = cv2.flip(results["img"], 1)  # horizontal flip

        # flip bboxes
        if results.get("bbox", None) is not None:
            results["bbox"] = self.flip_bbox(
                results["bbox"],
                image_size=(w, h),
                bbox_format="xyxy",
                direction=flip_dir,
            )

        if results.get("bbox_center", None) is not None:
            results["bbox_center"] = self.flip_bbox(
                results["bbox_center"],
                image_size=(w, h),
                bbox_format="center",
                direction=flip_dir,
            )

        # flip keypoints
        if results.get("keypoints", None) is not None:
            keypoints, keypoints_visible = self.flip_keypoints(
                results["keypoints"],
                results.get("keypoints_visible", None),
                image_size=(w, h),
                flip_indices=results["flip_indices"],
                direction=flip_dir,
            )

            results["keypoints"] = keypoints
            results["keypoints_visible"] = keypoints_visible

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"direction={self.direction})"
        return repr_str


@TRANSFORMS.register_module()
class PoseRandomHalfBody(BaseTransform):
    def __init__(
        self,
        min_total_keypoints: int = 9,
        min_upper_keypoints: int = 2,
        min_lower_keypoints: int = 3,
        padding: float = 1.5,
        prob: float = 0.3,
        upper_prioritized_prob: float = 0.7,
    ) -> None:
        super().__init__()
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def _get_half_body_bbox(
        self, keypoints: np.ndarray, half_body_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        selected_keypoints = keypoints[half_body_ids]
        center = selected_keypoints.mean(axis=0)[:2]
        x1, y1 = selected_keypoints.min(axis=0)
        x2, y2 = selected_keypoints.max(axis=0)
        w = x2 - x1
        h = y2 - y1
        scale = np.array([w, h], dtype=center.dtype) * self.padding
        return center, scale

    def _random_select_half_body(
        self,
        keypoints_visible: np.ndarray,
        upper_body_ids: List[int],
        lower_body_ids: List[int],
    ) -> List[Optional[List[int]]]:
        half_body_ids = []

        for visible in keypoints_visible:
            if visible.sum() < self.min_total_keypoints:
                indices = None
            elif np.random.rand() > self.prob:
                indices = None
            else:
                upper_valid_ids = [i for i in upper_body_ids if visible[i] > 0]
                lower_valid_ids = [i for i in lower_body_ids if visible[i] > 0]

                num_upper = len(upper_valid_ids)
                num_lower = len(lower_valid_ids)

                prefer_upper = np.random.rand() < self.upper_prioritized_prob
                if (
                    num_upper < self.min_upper_keypoints
                    and num_lower < self.min_lower_keypoints
                ):
                    indices = None
                elif num_lower < self.min_lower_keypoints:
                    indices = upper_valid_ids
                elif num_upper < self.min_upper_keypoints:
                    indices = lower_valid_ids
                else:
                    indices = upper_valid_ids if prefer_upper else lower_valid_ids

            half_body_ids.append(indices)

        return half_body_ids

    def transform(self, results: Dict) -> Optional[dict]:
        half_body_ids = self._random_select_half_body(
            keypoints_visible=results["keypoints_visible"],
            upper_body_ids=results["upper_body_ids"],
            lower_body_ids=results["lower_body_ids"],
        )

        bbox_center = []
        bbox_scale = []

        for i, indices in enumerate(half_body_ids):
            if indices is None:
                bbox_center.append(results["bbox_center"][i])
                bbox_scale.append(results["bbox_scale"][i])
            else:
                _center, _scale = self._get_half_body_bbox(
                    results["keypoints"][i], indices
                )
                bbox_center.append(_center)
                bbox_scale.append(_scale)

        results["bbox_center"] = np.stack(bbox_center)
        results["bbox_scale"] = np.stack(bbox_scale)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(min_total_keypoints={self.min_total_keypoints}, "
        repr_str += f"min_upper_keypoints={self.min_upper_keypoints}, "
        repr_str += f"min_lower_keypoints={self.min_lower_keypoints}, "
        repr_str += f"padding={self.padding}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"upper_prioritized_prob={self.upper_prioritized_prob})"
        return repr_str


@TRANSFORMS.register_module()
class PoseRandomBBoxTransform(BaseTransform):
    def __init__(
        self,
        shift_factor: float = 0.16,
        shift_prob: float = 0.3,
        scale_factor: Tuple[float, float] = (0.5, 1.5),
        scale_prob: float = 1.0,
        rotate_factor: float = 80.0,
        rotate_prob: float = 0.6,
    ) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(
        low: float = -1.0, high: float = 1.0, size: tuple = ()
    ) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        # Get shift parameters
        offset = self._truncnorm(size=(num_bboxes, 2)) * self.shift_factor
        offset = np.where(np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.0)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = self._truncnorm(size=(num_bboxes, 1)) * sigma + mu
        scale = np.where(np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.0)

        # Get rotation parameters
        rotate = self._truncnorm(size=(num_bboxes,)) * self.rotate_factor
        rotate = np.where(np.random.rand(num_bboxes) < self.rotate_prob, rotate, 0.0)

        return offset, scale, rotate

    def transform(self, results: Dict) -> Optional[dict]:
        bbox_scale = results["bbox_scale"]
        num_bboxes = bbox_scale.shape[0]

        offset, scale, rotate = self._get_transform_params(num_bboxes)

        results["bbox_center"] += offset * bbox_scale
        results["bbox_scale"] *= scale
        results["bbox_rotation"] = rotate

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(shift_prob={self.shift_prob}, "
        repr_str += f"shift_factor={self.shift_factor}, "
        repr_str += f"scale_prob={self.scale_prob}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"rotate_prob={self.rotate_prob}, "
        repr_str += f"rotate_factor={self.rotate_factor})"
        return repr_str


@TRANSFORMS.register_module()
class PoseAlbumentation(BaseTransform):
    def __init__(self, transforms: List[dict], keymap: Optional[dict] = None) -> None:
        if albumentations is None:
            raise RuntimeError("albumentations is not installed")
        self.transforms = transforms
        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms]
        )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
            }
        else:
            self.keymap_to_albu = keymap

    def albu_builder(self, cfg: dict) -> albumentations:
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            try:
                from torch.distributed import get_rank

                rank = get_rank()
            except (ImportError, RuntimeError):
                rank = 0
            obj_cls = getattr(albumentations, obj_type)
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]
        return obj_cls(**args)

    def transform(self, results: dict) -> dict:
        results_albu = {}
        for k, v in self.keymap_to_albu.items():
            assert k in results, (
                f"The `{k}` is required to perform albumentations transforms"
            )
            results_albu[v] = results[k]

        # Apply albumentations transforms
        results_albu = self.aug(**results_albu)

        # map the albu results back to the original format
        for k, v in self.keymap_to_albu.items():
            results[k] = results_albu[v]

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@TRANSFORMS.register_module()
class PosePackInputs(BaseTransform):
    def __init__(
        self,
        meta_keys=(
            "id",
            "img_id",
            "img_path",
            "category_id",
            "crowd_index",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
            "bbox_center",
            "bbox_scale",
            "bbox_score",
            "flip",
            "flip_direction",
            "flip_indices",
            "raw_ann_info",
        ),
        pack_transformed=False,
    ):
        self.meta_keys = meta_keys
        self.pack_transformed = pack_transformed

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = dict()
        if "keypoints" in results:
            keypoints = results["keypoints"].astype(np.float32)
            keypoints_visible = results["keypoints_visible"].astype(np.float32)
            data_sample["keypoints"] = keypoints
            data_sample["keypoints_visible"] = keypoints_visible

        ## update keypoints weights with if keypoints within bounds
        if "keypoint_weights" in results and "transformed_keypoints" in results:
            transformed_keypoints = results["transformed_keypoints"]  # 1 x K x 3
            h, w = img.shape[1:]

            keypoints_in_bounds = (
                keypoints_visible
                * (transformed_keypoints[..., 0] >= 0)
                * (transformed_keypoints[..., 1] >= 0)
                * (transformed_keypoints[..., 0] < w)
                * (transformed_keypoints[..., 1] < h)
            )
            data_sample["keypoint_weights"] = (
                keypoints_in_bounds * results["keypoint_weights"]
            )
        if "heatmaps" in results:
            data_sample["heatmaps"] = results["heatmaps"]  ## K x heatmap_H x heatmap_W

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                if isinstance(results[key], (int, float)):
                    img_meta[key] = np.float32(results[key])
                elif isinstance(results[key], np.ndarray):
                    img_meta[key] = results[key].astype(np.float32)
                else:
                    img_meta[key] = results[key]

        data_sample["meta"] = img_meta
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
