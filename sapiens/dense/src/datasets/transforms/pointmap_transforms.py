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
class PointmapRandomScale(BaseTransform):
    def __init__(
        self,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        prob: float = 0.5,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        super().__init__()
        assert 0 < scale_min <= scale_max, (
            f"Invalid scale range: ({scale_min}, {scale_max})"
        )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation
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
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
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

        if "gt_depth" in results:
            depth_resized = cv2.resize(
                results["gt_depth"],
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            results["gt_depth"] = depth_resized

        # 5. Update camera intrinsics if present
        if "K" in results:
            K_new = results["K"].copy()
            # Scale fx, fy
            K_new[0, 0] *= s  # fx
            K_new[1, 1] *= s  # fy
            # Shift principal point
            K_new[0, 2] *= s  # cx
            K_new[1, 2] *= s  # cy
            results["K"] = K_new

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale_min={self.scale_min}, "
            f"scale_max={self.scale_max})"
        )


@TRANSFORMS.register_module()
class PointmapRandomCrop(BaseTransform):
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
        # Pick one (h, w) from the list of possible crop sizes

        crop_h, crop_w = random.choice(self.crop_sizes)

        # Generate the crop bounding box
        crop_bbox = self._get_crop_bbox(img, crop_h, crop_w)

        # Apply to the main image
        cropped_img = self._crop_img(img, crop_bbox)

        results["img"] = cropped_img
        results["img_shape"] = cropped_img.shape[:2]

        # Crop other maps if they exist
        for key in ["gt_depth", "mask"]:
            if key in results:
                results[key] = self._crop_img(results[key], crop_bbox)

        # Adjust intrinsics if present
        if "K" in results:
            K_new = results["K"].copy()
            y1, y2, x1, x2 = crop_bbox
            # Shift principal point
            K_new[0, 2] -= x1
            K_new[1, 2] -= y1
            results["K"] = K_new

        return results


@TRANSFORMS.register_module()
class PointmapRandomCropContinuous(BaseTransform):
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
        for key in ["gt_depth", "mask"]:
            if key in results:
                results[key] = _crop(results[key], crop_bbox)

        # Adjust intrinsics if present
        if "K" in results:
            K_new = results["K"].copy()
            # Shift principal point
            y1, y2, x1, x2 = crop_bbox
            K_new[0, 2] -= x1
            K_new[1, 2] -= y1
            results["K"] = K_new

        return results


@TRANSFORMS.register_module()
class PointmapResize(BaseTransform):
    def __init__(self, height, width) -> None:
        super().__init__()
        self.target_height = height
        self.target_width = width

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
        if "mask" in results:
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

        if "gt_depth" in results:
            depth_resized = cv2.resize(
                results["gt_depth"],
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )
            final_depth = np.zeros(
                (self.target_height, self.target_width), dtype=depth_resized.dtype
            )
            final_depth[
                offset_y : offset_y + new_height, offset_x : offset_x + new_width
            ] = depth_resized
            results["gt_depth"] = final_depth

        # 9. Adjust camera intrinsics K accordingly
        if "K" in results:
            K_new = results["K"].copy()
            # Scale fx, fy
            K_new[0, 0] *= scale_factor  # fx
            K_new[1, 1] *= scale_factor  # fy

            # Scale and then shift principal point by offsets
            K_new[0, 2] = K_new[0, 2] * scale_factor + offset_x
            K_new[1, 2] = K_new[1, 2] * scale_factor + offset_y

            results["K"] = K_new

        return results


@TRANSFORMS.register_module()
class PointmapRandomFlip(BaseTransform):
    def __init__(self, prob=0.5) -> None:
        super().__init__()
        self.prob = prob

    def _flip(self, results: dict) -> None:
        """Flip images, masks, depth maps and adjust camera parameters."""
        # flip image
        results["img"] = cv2.flip(results["img"], 1)  # 1 for horizontal flip
        img_shape = results["img"].shape[:2]

        # flip seg map and depth (horizontal flip)
        results["mask"] = cv2.flip(results["mask"], 1)

        if "gt_depth" in results:
            results["gt_depth"] = cv2.flip(results["gt_depth"], 1)

        # adjust camera parameters
        if "K" in results:
            # Flip the principal point for the left-right flipped image
            results["K"][0, 2] = img_shape[1] - results["K"][0, 2] - 1

        if "M" in results:
            # Flip the sign of the first column of the extrinsics matrix
            results["M"][0, :] = -results["M"][0, :]

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if np.random.rand() < self.prob:
            self._flip(results)
        return results


@TRANSFORMS.register_module()
class PointmapGenerateTarget(BaseTransform):
    def __init__(self, canonical_focal_length=768, target_downsample_factor=None):
        self.canonical_focal_length = canonical_focal_length
        self.target_downsample_factor = target_downsample_factor
        return

    def transform(self, results: dict) -> dict:
        if "gt_depth" not in results.keys():
            return results

        ## only downsample gt_depth, mask and K
        if self.target_downsample_factor is not None:
            assert isinstance(self.target_downsample_factor, int)

            gt_depth = results["gt_depth"]
            mask = results["mask"]
            K = results["K"]

            gt_depth = cv2.resize(
                gt_depth,
                None,
                fx=1 / self.target_downsample_factor,
                fy=1 / self.target_downsample_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            mask = cv2.resize(
                mask,
                None,
                fx=1 / self.target_downsample_factor,
                fy=1 / self.target_downsample_factor,
                interpolation=cv2.INTER_NEAREST,
            )

            K[0, 0] = K[0, 0] / self.target_downsample_factor
            K[1, 1] = K[1, 1] / self.target_downsample_factor
            K[0, 2] = K[0, 2] / self.target_downsample_factor
            K[1, 2] = K[1, 2] / self.target_downsample_factor

            results["gt_depth"] = gt_depth
            results["mask"] = mask
            results["K"] = K

            if "uv_map" in results:
                uv_map = results["uv_map"]
                uv_map = cv2.resize(
                    uv_map,
                    None,
                    fx=1 / self.target_downsample_factor,
                    fy=1 / self.target_downsample_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
                results["uv_map"] = uv_map

        gt_depth = results["gt_depth"]  ## no normalization
        mask = results["mask"]

        fx = results["K"][0, 0]
        fy = results["K"][1, 1]
        cx = results["K"][0, 2]
        cy = results["K"][1, 2]

        scale = 1.0
        if self.canonical_focal_length is not None:
            scale = self.canonical_focal_length / fx

        cols, rows = np.meshgrid(
            np.arange(gt_depth.shape[1]), np.arange(gt_depth.shape[0])
        )
        X = (cols - cx) * gt_depth / fx
        Y = (rows - cy) * gt_depth / fy
        Z = gt_depth

        # # # ##-----------debug-----------------------
        # image = results['img']
        # K = results['K']
        # mask = results['mask'] > 0

        # # Set random seed
        # seed = np.random.randint(0, 10000)

        # # Project to image plane
        # x = K[0,0] * X/Z + K[0,2]  # new_fx * X/Z + cx
        # y = K[1,1] * Y/Z + K[1,2]  # new_fy * Y/Z + cy

        # # Round to nearest pixel and clip to image bounds
        # x = np.clip(np.round(x), 0, image.shape[1]-1).astype(int)
        # y = np.clip(np.round(y), 0, image.shape[0]-1).astype(int)

        # # Create visualization
        # debug_img = image.copy()
        # # Draw all valid projected points in green
        # debug_img[y[mask], x[mask]] = [0, 255, 0]  # Set projected points to green
        # debug_img = np.concatenate([image, debug_img], axis=1)

        # # Save debug image
        # cv2.imwrite(f'seed{seed}.jpg', debug_img)
        # # -----------------------------------------

        # Scale the coordinates. isotropic scaling
        X = X * scale
        Y = Y * scale
        Z = Z * scale

        results["original_K"] = results["K"].copy()
        results["scale"] = scale

        if self.canonical_focal_length is not None:
            # New camera intrinsics
            new_K = results["K"].copy()
            new_K[0, 0] = fx * scale  # new fx
            new_K[1, 1] = fy * scale  # new fy
            new_K[0, 2] = cx * scale
            new_K[1, 2] = cy * scale
            results["K"] = new_K

        gt_pointmap = np.stack([X, Y, Z], axis=-1)
        results["gt_depth"] = Z  ## canonical depth

        ## preserve range by removing invalid points
        gt_pointmap[mask == 0] = 0
        results["gt_pointmap"] = gt_pointmap

        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class PointmapPackInputs(BaseTransform):
    def __init__(
        self,
        meta_keys=(
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "K",
            "original_K",
            "M",
        ),
    ):
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

        if "gt_pointmap" in results:
            mask = results["mask"] > 0  ## boolean mask

            ## min number of valid pixels is 4
            if mask.sum() < 16:
                return None

            if mask.sum() / (mask.shape[0] * mask.shape[1]) > 0.96:
                return None

            ## clipping inside camera
            min_depth = results["gt_pointmap"][results["mask"] > 0, 2].min()
            if min_depth < 0.04:
                return None

            gt_mean_depth = results["gt_pointmap"][results["mask"] > 0, 2].mean()

            # ##------------------debug------------------
            # ## print the min, max and mean of the X, Y, Z coordinates
            # X = results["gt_pointmap"][results["mask"] > 0, 0]
            # Y = results["gt_pointmap"][results["mask"] > 0, 1]
            # Z = results["gt_pointmap"][results["mask"] > 0, 2]
            # inv_Z = 1 / Z

            # print("scale:", results["scale"])
            # print("X min:", X.min(), "X max:", X.max(), "X mean:", X.mean())
            # print("Y min:", Y.min(), "Y max:", Y.max(), "Y mean:", Y.mean())
            # print("Z min:", Z.min(), "Z max:", Z.max(), "Z mean:", Z.mean())
            # print(
            #     "inv_Z min:",
            #     inv_Z.min(),
            #     "inv_Z max:",
            #     inv_Z.max(),
            #     "inv_Z mean:",
            #     inv_Z.mean(),
            # )
            # print()
            ##-----------------------------------------
            mask = to_tensor(mask[None, ...].copy())  ## 1 x H x W
            data_sample["mask"] = mask

            gt_pointmap = results["gt_pointmap"].astype(np.float32)  ## H x W x 3
            gt_pointmap = gt_pointmap.transpose(2, 0, 1)  ## H x W x 3 -> 3 x H x W
            data_sample["gt_pointmap"] = to_tensor(gt_pointmap.copy())
            data_sample["gt_mean_depth"] = to_tensor(
                gt_mean_depth[None, None, None].copy()
            )

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
class PointmapResizePadImage(BaseTransform):
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
