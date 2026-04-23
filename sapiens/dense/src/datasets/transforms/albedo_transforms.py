# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sapiens.engine.datasets import BaseTransform, to_tensor
from sapiens.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AlbedoRandomScale(BaseTransform):
    def __init__(
        self,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        prob: float = 0.5,
    ):
        super().__init__()
        assert 0 < scale_min <= scale_max, (
            f"Invalid scale range: ({scale_min}, {scale_max})"
        )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.prob = prob

    def _random_scale_factor(self) -> float:
        """Sample a random scale factor in [scale_min, scale_max]."""
        return np.random.uniform(self.scale_min, self.scale_max)

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results

        img = results["img"]
        orig_h, orig_w = img.shape[:2]

        # 1. Sample a random scale factor
        s = self._random_scale_factor()

        # 2. Compute the new size
        new_w = int(round(orig_w * s))
        new_h = int(round(orig_h * s))

        # 3. Resize the image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results["img"] = img_resized
        results["img_shape"] = (new_h, new_w)

        # 4. Resize mask, depth, etc. using INTER_NEAREST
        if "mask" in results:
            mask_resized = cv2.resize(
                results["mask"].astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            results["mask"] = mask_resized

        if "gt_albedo" in results:
            albedo_resized = cv2.resize(
                results["gt_albedo"], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            results["gt_albedo"] = albedo_resized

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale_min={self.scale_min}, "
            f"scale_max={self.scale_max})"
        )


@TRANSFORMS.register_module()
class AlbedoRandomCrop(BaseTransform):
    def __init__(self, crop_sizes: List[Tuple[int, int]], prob: float = 0.5):
        super().__init__()
        assert isinstance(crop_sizes, list) and len(crop_sizes) > 0, (
            "crop_sizes must be a non-empty list of (h, w) tuples."
        )
        for size in crop_sizes:
            assert len(size) == 2 and size[0] > 0 and size[1] > 0, (
                f"Invalid crop size: {size}"
            )

        self.crop_sizes = crop_sizes
        self.prob = prob

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(crop_sizes={self.crop_sizes}, prob={self.prob})"
        )

    def _get_crop_bbox(self, img: np.ndarray, crop_h: int, crop_w: int) -> tuple:
        """Randomly generate a crop bounding box for an image given target (h, w)."""
        h, w = img.shape[:2]

        # Ensure the target crop is not bigger than the image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        margin_h = h - crop_h
        margin_w = w - crop_w

        # Random top-left corner
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        y1, y2 = offset_h, offset_h + crop_h
        x1, x2 = offset_w, offset_w + crop_w
        return (y1, y2, x1, x2)

    def _crop_img(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop image to bbox = (y1, y2, x1, x2)."""
        y1, y2, x1, x2 = crop_bbox
        return img[y1:y2, x1:x2, ...]

    def transform(self, results: dict) -> dict:
        # Decide whether to apply cropping
        if np.random.rand() >= self.prob:
            return results  # skip cropping

        img = results["img"]
        crop_h, crop_w = random.choice(self.crop_sizes)
        crop_bbox = self._get_crop_bbox(img, crop_h, crop_w)
        cropped_img = self._crop_img(img, crop_bbox)

        results["img"] = cropped_img
        results["img_shape"] = cropped_img.shape[:2]

        # Crop other maps if they exist
        for key in ["gt_albedo", "mask"]:
            if key in results:
                results[key] = self._crop_img(results[key], crop_bbox)

        return results


@TRANSFORMS.register_module()
class AlbedoRandomCropContinuous(BaseTransform):
    def __init__(
        self,
        ar_range: Tuple[float, float] = (0.5, 2.0),
        area_range: Tuple[float, float] = (0.1, 1.0),
        num_attempts: int = 10,
        prob: float = 0.5,
    ):
        super().__init__()
        assert ar_range[0] > 0 and ar_range[1] >= ar_range[0], (
            f"Invalid ar_range={ar_range}"
        )
        assert area_range[0] > 0 and area_range[1] >= area_range[0], (
            f"Invalid area_range={area_range}"
        )
        self.ar_range = ar_range
        self.area_range = area_range
        self.num_attempts = num_attempts
        self.prob = prob

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ar_range={self.ar_range}, "
            f"area_range={self.area_range}, "
            f"num_attempts={self.num_attempts}, "
            f"prob={self.prob})"
        )

    def transform(self, results: Dict) -> Dict:
        """Apply the random aspect-ratio crop if conditions are met."""
        if not (random.random() < self.prob):
            return results  # skip cropping

        img = results["img"]
        orig_h, orig_w = img.shape[:2]
        img_area = orig_h * orig_w

        # Try up to num_attempts times to find a valid crop
        for attempt in range(self.num_attempts):
            # 1) Sample aspect ratio in [ar_min, ar_max]
            ar = random.uniform(*self.ar_range)  # aspect ratio

            # 2) Sample area fraction in [area_min, area_max]
            area_frac = random.uniform(*self.area_range)
            target_area = area_frac * img_area

            # 3) Solve for crop_h, crop_w
            crop_h = math.sqrt(target_area / ar)
            crop_w = ar * crop_h

            # 4) Check feasibility: both must be <= orig dims
            if crop_w <= orig_w and crop_h <= orig_h:
                # 5) Random top-left corner
                crop_h = int(round(crop_h))
                crop_w = int(round(crop_w))
                margin_h = orig_h - crop_h
                margin_w = orig_w - crop_w
                y1 = random.randint(0, margin_h + 1)
                x1 = random.randint(0, margin_w + 1)

                y2 = y1 + crop_h
                x2 = x1 + crop_w

                # We found a valid crop
                crop_bbox = (y1, y2, x1, x2)
                break
        else:
            # If we never broke out, no valid crop found; skip
            # (or we could do a fallback like no crop)
            return results

        # --- We do the actual cropping now ---
        def _crop(img_: np.ndarray, bbox: tuple) -> np.ndarray:
            (yy1, yy2, xx1, xx2) = bbox
            return img_[yy1:yy2, xx1:xx2, ...]

        # Crop the main image
        cropped_img = _crop(img, crop_bbox)
        results["img"] = cropped_img
        results["img_shape"] = cropped_img.shape[:2]

        # Crop depth/mask if present
        for key in ["gt_albedo", "mask"]:
            if key in results:
                results[key] = _crop(results[key], crop_bbox)

        return results


@TRANSFORMS.register_module()
class AlbedoResize(BaseTransform):
    def __init__(self, height, width, test_mode: bool = False) -> None:
        super().__init__()
        self.target_height = height
        self.target_width = width
        self.test_mode = test_mode

    def transform(self, results: Dict) -> Dict:
        img = results["img"]
        orig_height, orig_width = img.shape[:2]

        # 1. Compute the scale factor to maintain aspect ratio
        scale_w = self.target_width / orig_width
        scale_h = self.target_height / orig_height
        scale_factor = min(scale_w, scale_h)

        # 2. Determine new (width, height) after aspect-preserving resize
        new_width = int(round(orig_width * scale_factor))
        new_height = int(round(orig_height * scale_factor))

        # 3. Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # 4. Create a black canvas of final size [H, W]
        final_img = np.zeros(
            (self.target_height, self.target_width, resized_img.shape[2])
            if resized_img.ndim == 3
            else (self.target_height, self.target_width),
            dtype=resized_img.dtype,
        )

        # 5. Compute offsets to center the resized image
        offset_x = (self.target_width - new_width) // 2
        offset_y = (self.target_height - new_height) // 2

        # 6. Copy resized image into the canvas
        if final_img.ndim == 3:  # color image
            final_img[
                offset_y : offset_y + new_height, offset_x : offset_x + new_width, :
            ] = resized_img
        else:  # single-channel image
            final_img[
                offset_y : offset_y + new_height, offset_x : offset_x + new_width
            ] = resized_img

        # 7. Replace `results['img']` with our padded image
        results["img"] = final_img
        results["img_shape"] = final_img.shape[:2]

        # 8. Do the same for mask & gt_depth
        #    (using nearest interpolation, then padding to center)
        if "mask" in results and self.test_mode is False:
            mask_resized = cv2.resize(
                results["mask"].astype(np.uint8),
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )
            final_mask = np.zeros(
                (self.target_height, self.target_width), dtype=mask_resized.dtype
            )
            final_mask[
                offset_y : offset_y + new_height, offset_x : offset_x + new_width
            ] = mask_resized
            results["mask"] = final_mask

        if "gt_albedo" in results and self.test_mode is False:
            albedo_resized = cv2.resize(
                results["gt_albedo"],
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            final_albedo = np.zeros(
                (self.target_height, self.target_width, 3), dtype=albedo_resized.dtype
            )
            final_albedo[
                offset_y : offset_y + new_height, offset_x : offset_x + new_width, :
            ] = albedo_resized
            results["gt_albedo"] = final_albedo

        return results


@TRANSFORMS.register_module()
class AlbedoRandomFlip(BaseTransform):
    def __init__(self, prob=0.5) -> None:
        super().__init__()
        self.prob = prob

    def _flip(self, results: dict) -> None:
        """Flip images, masks, depth maps and adjust camera parameters."""
        # flip image
        results["img"] = cv2.flip(results["img"], 1)  # 1 for horizontal flip

        # flip seg map and depth (horizontal flip)
        results["mask"] = cv2.flip(results["mask"], 1)

        if "gt_albedo" in results:
            gt_albedo = results["gt_albedo"]
            gt_albedo = cv2.flip(gt_albedo, 1)  # 1 for horizontal flip
            results["gt_albedo"] = gt_albedo

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if np.random.rand() < self.prob:
            self._flip(results)
        return results


@TRANSFORMS.register_module()
class AlbedoPackInputs(BaseTransform):
    def __init__(
        self,
        test_mode: bool = False,
        meta_keys=(
            "img_path",
            "ori_shape",
            "img_shape",
        ),
    ):
        self.test_mode = test_mode
        self.meta_keys = meta_keys

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

        if "gt_albedo" in results:
            mask = results["mask"] > 0  ## boolean mask

            ## min number of valid pixels is 16
            if mask.sum() < 16 and self.test_mode is False:
                return None

            if (mask.sum() / (mask.shape[0] * mask.shape[1]) > 0.96) and (
                self.test_mode is False
            ):
                return None

            ##-----------------------------------------
            mask = to_tensor(mask[None, ...].copy())  ## 1 x H x W
            data_sample["mask"] = mask

            gt_albedo = results["gt_albedo"].astype(np.float32)  ## H x W x 3
            gt_albedo = gt_albedo.transpose(2, 0, 1)  # H x W x 3 -> 3 x H x W
            data_sample["gt_albedo"] = to_tensor(gt_albedo.copy())

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


@TRANSFORMS.register_module()
class AlbedoResizePadImage(BaseTransform):
    def __init__(
        self,
        height: int = 1024,
        width: int = 768,
        pad_val: Optional[int] = 0,
        padding_mode: str = "constant",
    ) -> None:
        self.height = height
        self.width = width
        self.pad_val = pad_val
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.padding_mode = padding_mode

    def _resize_maintain_aspect_ratio(self, img, target_size):
        """Resize image maintaining aspect ratio and return padding sizes."""
        original_height, original_width = img.shape[:2]
        target_width, target_height = target_size

        # Calculate scaling factors
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)  # Use the smaller scaling factor

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding
        pad_width = target_width - new_width
        pad_height = target_height - new_height

        padding_left = pad_width // 2
        padding_right = pad_width - padding_left
        padding_top = pad_height // 2
        padding_bottom = pad_height - padding_top

        return resized_img, (padding_left, padding_right, padding_top, padding_bottom)

    def _pad_img(self, results: dict) -> None:
        """Resize image maintaining aspect ratio and pad to target size."""
        img = results["img"]
        target_size = (self.width, self.height)  # (width, height)

        # Resize image maintaining aspect ratio
        resized_img, padding_size = self._resize_maintain_aspect_ratio(img, target_size)

        # Prepare padding value
        pad_val = self.pad_val

        # Pad image
        padding_left, padding_right, padding_top, padding_bottom = padding_size
        if resized_img.ndim == 3:
            padded_img = np.pad(
                resized_img,
                ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                mode=self.padding_mode,
                constant_values=pad_val,
            )
        else:
            padded_img = np.pad(
                resized_img,
                ((padding_top, padding_bottom), (padding_left, padding_right)),
                mode=self.padding_mode,
                constant_values=pad_val,
            )

        # Update results dictionary
        results["img"] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = target_size
        results["img_shape"] = padded_img.shape[:2]
        results["padding_size"] = padding_size

    def transform(self, results: dict) -> dict:
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"height={self.height}, "
        repr_str += f"width={self.width}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"padding_mode={self.padding_mode})"
        return repr_str
