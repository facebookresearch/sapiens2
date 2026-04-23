# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import warnings

import cv2
import numpy as np
from sapiens.engine.datasets import BaseTransform, to_tensor
from sapiens.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SegRandomRotate(BaseTransform):
    def __init__(
        self,
        prob=0.5,
        degree=60,
        pad_val=0,
        seg_pad_val=255,
    ):
        super().__init__()
        self.prob = prob
        assert prob >= 0 and prob <= 1
        assert degree > 0, f"degree {degree} should be positive"
        self.degree = (-degree, degree)
        assert len(self.degree) == 2, (
            f"degree {self.degree} should be a tuple of (min, max)"
        )
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        degree = random.uniform(min(*self.degree), max(*self.degree))
        img = results["img"]
        gt_seg = results["gt_seg"]
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, degree, 1.0)

        results["img"] = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=self.pad_val
        )

        # Rotate the segmentation map
        results["gt_seg"] = cv2.warpAffine(
            gt_seg,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=self.seg_pad_val,
        )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(prob={self.prob}, "
            f"degree={self.degree}, "
            f"pad_val={self.pal_val}, "
            f"seg_pad_val={self.seg_pad_val}, "
        )
        return repr_str


@TRANSFORMS.register_module()
class SegRandomHorizontalFlip(BaseTransform):
    def __init__(self, prob=0.5, swap_seg_labels=None):
        super().__init__()
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        img = results["img"]
        gt_seg = results["gt_seg"]
        img = cv2.flip(img, 1)
        gt_seg = cv2.flip(gt_seg, 1)
        temp = gt_seg.copy()
        if self.swap_seg_labels is not None:
            for pair in self.swap_seg_labels:
                assert len(pair) == 2
                gt_seg[temp == pair[0]] = pair[1]
                gt_seg[temp == pair[1]] = pair[0]

        results["img"] = img
        results["gt_seg"] = gt_seg
        return results


@TRANSFORMS.register_module()
class SegPackInputs(BaseTransform):
    def __init__(
        self,
        test_mode: bool = False,
        meta_keys=(
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "flip",
        ),
    ):
        super().__init__()
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

        if "gt_seg" in results:
            assert len(results["gt_seg"].shape) == 2  # H x W
            mask = (results["gt_seg"] > 0) * (results["gt_seg"] != 255)
            if (
                mask.sum() / (mask.shape[0] * mask.shape[1]) < 0.01
                and self.test_mode == False
            ):
                return None

            data_sample["gt_seg"] = to_tensor(
                results["gt_seg"][None, ...].astype(np.int64)
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
class SegRandomResize(BaseTransform):
    def __init__(
        self,
        base_height=1024,
        base_width=768,
        ratio_range=(0.4, 2.0),
        keep_ratio=True,
    ):
        super().__init__()
        self.base_height = base_height
        self.base_width = base_width
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.resizer = SegResize(
            height=self.base_height, width=self.base_width, keep_ratio=keep_ratio
        )

    def transform(self, results: dict) -> dict:
        min_ratio, max_ratio = self.ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        self.resizer.height = int(self.base_height * ratio)
        self.resizer.width = int(self.base_width * ratio)
        return self.resizer.transform(results)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"base_height={self.base_height}, "
            f"base_width={self.base_width}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )


@TRANSFORMS.register_module()
class SegResize(BaseTransform):
    def __init__(
        self,
        height=1024,
        width=768,
        keep_ratio=False,
        test_mode: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.keep_ratio = keep_ratio
        self.test_mode = test_mode

    def transform(self, results: dict) -> dict:
        img = results["img"]
        h, w = img.shape[:2]

        target_height = self.height
        target_width = self.width

        if self.keep_ratio is True:
            scale_factor = min(target_width / w, target_height / h)
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))

        else:
            new_w = target_width
            new_h = target_height

        dsize = (new_w, new_h)

        # Use INTER_AREA for shrinking and INTER_CUBIC for enlarging
        # to get antialiased results.
        img_interpolation = cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC

        # Update the results dictionary
        results["img"] = cv2.resize(img, dsize, interpolation=img_interpolation)

        ## resize gt seg if training
        if "gt_seg" in results and self.test_mode is False:
            gt_seg = results["gt_seg"]
            results["gt_seg"] = cv2.resize(
                gt_seg, dsize, interpolation=cv2.INTER_NEAREST
            )
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"height={self.height}, "
            f"width={self.width}, "
            f"keep_ratio={self.keep_ratio})"
        )


@TRANSFORMS.register_module()
class SegRandomCrop(BaseTransform):
    def __init__(
        self,
        crop_height=1024,
        crop_width=768,
        prob=0.5,
        cat_max_ratio=0.75,
        ignore_index=255,
    ):
        super().__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.prob = prob
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def _generate_crop_bbox(self, img: np.ndarray) -> tuple:
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_height, 0)
        margin_w = max(img.shape[1] - self.crop_width, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_height
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_width
        return crop_y1, crop_y2, crop_x1, crop_x2

    def _crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images and segmentation maps."""
        if random.random() > self.prob:
            return results

        img = results["img"]
        gt_seg = results["gt_seg"]
        h, w = img.shape[:2]

        # Pad the image if it's smaller than the crop size
        pad_h = max(self.crop_height - h, 0)
        pad_w = max(self.crop_width - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(
                img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
            gt_seg = cv2.copyMakeBorder(
                gt_seg, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index
            )

        padded_img = img
        padded_gt_seg = gt_seg

        crop_bbox = self._generate_crop_bbox(padded_img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times to find a valid crop
            for _ in range(10):
                seg_temp = self._crop(padded_gt_seg, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                # Filter out the ignore_index
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break  # Found a valid crop
                crop_bbox = self._generate_crop_bbox(padded_img)

        # Crop the image and segmentation map
        results["img"] = self._crop(padded_img, crop_bbox)
        results["gt_seg"] = self._crop(padded_gt_seg, crop_bbox)

        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"crop_height={self.crop_height}, "
            f"crop_width={self.crop_width}, "
            f"prob={self.prob}, "
            f"cat_max_ratio={self.cat_max_ratio}, "
            f"ignore_index={self.ignore_index})"
        )


@TRANSFORMS.register_module()
class SegRandomBackground(BaseTransform):
    def __init__(
        self,
        prob: float = 0.25,
        skip_key: str = "is_itw",
        background_images_root: str = "",
    ):
        super().__init__()

        self.prob = prob
        self.skip_key = skip_key
        self.background_images_root = background_images_root
        self.background_images = sorted(
            [
                image_name
                for image_name in os.listdir(background_images_root)
                if image_name.endswith(".jpg")
            ]
        )

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        if self.skip_key in results and results[self.skip_key]:
            return results

        image = results["img"]  ## bgr image

        if "gt_seg" in results:
            gt_seg = results["gt_seg"]
            mask = (gt_seg > 0).astype(np.uint8)

        elif "mask" in results:
            mask = results["mask"]
            mask = (mask > 0).astype(np.uint8)

        else:
            warnings.warn(
                f"foreground mask not found in results, skip random background!"
            )
            return results

        background_image_path = os.path.join(
            self.background_images_root, random.choice(self.background_images)
        )
        background_image = cv2.imread(background_image_path)  ## bgr image

        ##-----------------------------
        background_height = background_image.shape[0]
        background_width = background_image.shape[1]

        image_height = image.shape[0]
        image_width = image.shape[1]

        new_background_height = image_height
        new_background_width = int(
            new_background_height * background_width / background_height
        )
        background_image = cv2.resize(
            background_image, (new_background_width, new_background_height)
        )

        # Crop the background image to the width of the original image
        if new_background_width > image_width:
            start_x = (new_background_width - image_width) // 2
            end_x = start_x + image_width
            background_image = background_image[:, start_x:end_x]

        if (
            background_image.shape[0] != image_height
            or background_image.shape[1] != image_width
        ):
            background_image = cv2.resize(background_image, (image_width, image_height))

        # Use the segmentation mask as an alpha channel.
        alpha_norm = mask.astype(np.float32)  # Values 0 or 1.
        alpha_mask = np.stack([alpha_norm] * 3, axis=-1)
        composite = alpha_mask * image + (1 - alpha_mask) * background_image
        composite = composite.astype(np.uint8)

        # Apply color transfer using the Reinhard algorithm.
        composite = self.reinhard_alpha(composite, alpha_norm)
        results["img"] = composite

        return results

    def reinhard_alpha(self, comp_img, alpha_mask):
        """
        # Reinhard color transfer algorithm with alpha mask support
        # paper: https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
        # alpha mask in range [0, 1]
        """

        # Convert to LAB color space
        comp_lab = cv2.cvtColor(comp_img, cv2.COLOR_BGR2Lab)

        # Calculate weighted mean and std for background and foreground
        bg_weights = 1 - alpha_mask
        fg_weights = alpha_mask

        bg_mean, bg_std = self.weighted_mean_std(comp_lab, bg_weights)
        fg_mean, fg_std = self.weighted_mean_std(comp_lab, fg_weights)

        # Avoid division by zero
        fg_std = np.maximum(fg_std, 1e-6)

        ratio = (bg_std / fg_std).reshape(-1)
        offset = (bg_mean - fg_mean * bg_std / fg_std).reshape(-1)

        # Apply color transfer
        trans_lab = cv2.convertScaleAbs(comp_lab * ratio + offset)
        trans_img = cv2.cvtColor(trans_lab, cv2.COLOR_Lab2BGR)

        # Blend the transferred image with the original image using the alpha mask
        alpha_mask_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        trans_comp = (
            trans_img * alpha_mask_3d + comp_img * (1 - alpha_mask_3d)
        ).astype(np.uint8)

        return trans_comp

    def weighted_mean_std(self, img, weights):
        # Ensure weights have the same shape as img
        weights_3d = np.repeat(weights[:, :, np.newaxis], img.shape[2], axis=2)

        # Calculate weighted mean
        total_weights = np.sum(weights_3d, axis=(0, 1))
        mean = np.sum(img * weights_3d, axis=(0, 1)) / total_weights

        # Calculate weighted standard deviation
        variance = np.sum(((img - mean) ** 2) * weights_3d, axis=(0, 1)) / total_weights
        std = np.sqrt(variance)

        return mean, std
