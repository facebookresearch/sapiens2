# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torchvision.transforms as T
from sapiens.registry import TRANSFORMS

from .base_transform import BaseTransform, to_tensor


@TRANSFORMS.register_module()
class ImageResize(BaseTransform):
    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.image_height = image_height
        self.image_width = image_width

    def transform(self, results: Dict) -> Optional[Dict]:
        image = results["image"]
        image = cv2.resize(
            image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA
        )
        results["image"] = image

        return results


@TRANSFORMS.register_module()
class ImagePackInputs(BaseTransform):
    def __init__(self, meta_keys: List[str]):
        self.meta_keys = meta_keys
        self.to_tensor = T.ToTensor()

    def transform(self, results: Dict) -> Optional[Dict]:
        packed_results = dict()

        raw_image = results["image"]
        image = raw_image.copy()
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if not image.flags.c_contiguous:
            image = to_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            image = image.transpose(2, 0, 1)
            image = to_tensor(image).contiguous()
        packed_results["inputs"] = image

        data_samples = dict()

        # Pack the specified meta keys
        for key in self.meta_keys:
            if key in results:
                data_samples[key] = results[key]

        data_samples["image"] = self.to_tensor(raw_image)
        packed_results["data_samples"] = data_samples

        return packed_results


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Sequence[float] = (0.5, 1.5),
        saturation_range: Sequence[float] = (0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img: np.ndarray, alpha: int = 1, beta: int = 0) -> np.ndarray:
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            return self.convert(
                img, alpha=random.uniform(self.contrast_lower, self.contrast_upper)
            )
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper),
            )
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int)
                + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def transform(self, results: dict) -> dict:
        img = results["img"]
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@TRANSFORMS.register_module()
class RandomPhotoMetricDistortion(PhotoMetricDistortion):
    def __init__(
        self,
        prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prob = prob

    def transform(self, results: Dict) -> Optional[Dict]:
        if np.random.rand() > self.prob:
            return results
        return super().transform(results)


@TRANSFORMS.register_module()
class RandomDownUpSampleImage(BaseTransform):
    _INTERP_LIST = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ]

    def __init__(self, scale_range=(0.1, 0.5), prob=0.4):
        super().__init__()
        self.scale_range = scale_range
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results  # Skip with probability (1 - prob)

        img = results["img"]
        orig_h, orig_w = img.shape[:2]

        # Pick a random factor in [min_scale, max_scale]
        min_scale, max_scale = self.scale_range
        scale_factor = np.random.uniform(min_scale, max_scale)

        # Randomly select interpolation modes for downsampling and upsampling
        down_interp = random.choice(self._INTERP_LIST)
        up_interp = random.choice(self._INTERP_LIST)

        # Compute downsample size
        down_w = max(1, int(orig_w * scale_factor))
        down_h = max(1, int(orig_h * scale_factor))

        # Downsample
        img_down = cv2.resize(img, (down_w, down_h), interpolation=down_interp)
        img_up = cv2.resize(img_down, (orig_w, orig_h), interpolation=up_interp)

        # Replace the original image with the heavily down-up-sampled version
        results["img"] = img_up
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale_range={self.scale_range}, "
            f"prob={self.prob})"
        )


@TRANSFORMS.register_module()
class RandomGaussianBlur(BaseTransform):
    def __init__(self, prob=0.4, kernel_size=(3, 3), sigma_range=(0.1, 2.0)):
        super().__init__()
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        if self.sigma_range is not None:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        else:
            sigma = 0  # OpenCV auto-calculates

        blurred = cv2.GaussianBlur(img, self.kernel_size, sigma)
        results["img"] = blurred
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"kernel_size={self.kernel_size}, sigma_range={self.sigma_range})"
        )


@TRANSFORMS.register_module()
class RandomJPEGCompression(BaseTransform):
    def __init__(self, prob=0.4, quality_range=(30, 60)):
        super().__init__()
        self.prob = prob
        self.quality_range = quality_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        q_min, q_max = self.quality_range
        quality = np.random.randint(q_min, q_max + 1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc_img = cv2.imencode(".jpg", img, encode_param)
        if success:
            dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
            results["img"] = dec_img
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"quality_range={self.quality_range})"
        )


@TRANSFORMS.register_module()
class RandomGaussianNoise(BaseTransform):
    def __init__(self, prob=0.4, mean=0.0, var_range=(5.0, 20.0)):
        super().__init__()
        self.prob = prob
        self.mean = mean
        self.var_range = var_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"].astype(np.float32)
        var = np.random.uniform(self.var_range[0], self.var_range[1])
        sigma = var**0.5

        noise = np.random.normal(self.mean, sigma, img.shape).astype(np.float32)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        results["img"] = noisy_img
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"mean={self.mean}, var_range={self.var_range})"
        )


@TRANSFORMS.register_module()
class RandomGamma(BaseTransform):
    def __init__(self, prob=0.4, gamma_range=(0.7, 1.3)):
        super().__init__()
        self.prob = prob
        self.gamma_range = gamma_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])

        # Build a lookup table for [0..255]
        table = (
            np.array([(i / 255.0) ** gamma * 255 for i in range(256)])
            .clip(0, 255)
            .astype(np.uint8)
        )
        img_corrected = cv2.LUT(img, table)
        results["img"] = img_corrected
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"gamma_range={self.gamma_range})"
        )


@TRANSFORMS.register_module()
class RandomGrayscale(BaseTransform):
    def __init__(self, prob=0.4):
        super().__init__()
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results["img"] = gray_3ch
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"


@TRANSFORMS.register_module()
class RandomChannelShuffle(BaseTransform):
    def __init__(self, prob=0.4):
        super().__init__()
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        channels = [0, 1, 2]
        np.random.shuffle(channels)
        img = img[..., channels]
        results["img"] = img
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"


@TRANSFORMS.register_module()
class RandomInvert(BaseTransform):
    def __init__(self, prob=0.4):
        super().__init__()
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        results["img"] = 255 - img
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"


@TRANSFORMS.register_module()
class RandomSolarize(BaseTransform):
    def __init__(self, prob=0.4, threshold=128):
        super().__init__()
        self.prob = prob
        self.threshold = threshold

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        mask = img > self.threshold
        img[mask] = 255 - img[mask]
        results["img"] = img
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, threshold={self.threshold})"
        )


@TRANSFORMS.register_module()
class RandomPosterize(BaseTransform):
    def __init__(self, prob=0.4, bits=(2, 5)):
        super().__init__()
        self.prob = prob
        self.bits = bits

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        # pick random bits
        bits_chosen = random.randint(self.bits[0], self.bits[1])
        shift = 8 - bits_chosen
        img = (img >> shift) << shift
        results["img"] = img
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob}, bits={self.bits})"
