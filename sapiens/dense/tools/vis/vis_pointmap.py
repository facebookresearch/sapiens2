# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError(
        "open3d is required for pointmap visualization. "
        "Install with: pip install open3d  (or `pip install -e .[pointmap]`)"
    ) from e
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TVF
from matplotlib import pyplot as plt
from sapiens.dense.models import init_model
from tqdm import tqdm

cmap = plt.get_cmap("turbo")
torchvision.disable_beta_transforms_warning()


# -------------------------------------------------------------------------------
def process_depth_map_with_bounds(depth_map, mask, min_val, max_val):
    """Render depth as turbo colormap using pre-computed (global) min/max bounds."""
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

    if min_val is None or max_val is None or not np.any(mask):
        return processed_depth

    depth_foreground = depth_map[mask > 0]
    if len(depth_foreground) == 0:
        return processed_depth

    inverse_depth_foreground = 1 / np.clip(depth_foreground, 1e-6, None)
    max_inverse_depth = min(1 / max(min_val, 1e-6), 1 / 0.1)
    min_inverse_depth = max(1 / 250, 1 / max(max_val, 1e-6))

    inverse_depth_foreground_normalized = (
        inverse_depth_foreground - min_inverse_depth
    ) / (max_inverse_depth - min_inverse_depth)
    inverse_depth_foreground_normalized = np.clip(
        inverse_depth_foreground_normalized, 0, 1
    )

    color_depth = (cmap(inverse_depth_foreground_normalized)[..., :3] * 255).astype(
        np.uint8
    )
    processed_depth[mask] = color_depth
    processed_depth = processed_depth[..., ::-1]  ## convert RGB to BGR to save with cv2
    return np.array(processed_depth, dtype=np.uint8)


def compute_surface_normals(depth_map, mask, min_val, max_val, kernel_size=7):
    """Compute surface normals from depth map."""
    depth_normalized = np.full(mask.shape, np.inf)
    depth_normalized[mask > 0] = 1 - (
        (depth_map[mask > 0] - min_val) / (max_val - min_val)
    )

    grad_x = cv2.Sobel(
        depth_normalized.astype(np.float32), cv2.CV_32F, 1, 0, ksize=kernel_size
    )
    grad_y = cv2.Sobel(
        depth_normalized.astype(np.float32), cv2.CV_32F, 0, 1, ksize=kernel_size
    )
    normals = np.dstack((-grad_x, -grad_y, np.full(grad_x.shape, -1)))

    normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_normalized = normals / (normals_mag + 1e-5)
    normal_vis = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)
    return normal_vis[:, :, ::-1]


def resize_pointmap(
    pointmap,
    target_height,
    target_width,
    smooth=False,
    blur_ks=3,
    blur_sigma=0.8,
    smooth_iters=4,
):
    assert pointmap.dim() == 4 and pointmap.shape[1] == 3, "pointmap must be 1x3xHxW"
    H, W = pointmap.shape[2], pointmap.shape[3]
    up = (target_height > H) or (target_width > W)

    if smooth and up:
        for _ in range(smooth_iters):
            pointmap = TVF.gaussian_blur(
                pointmap, kernel_size=[blur_ks, blur_ks], sigma=[blur_sigma, blur_sigma]
            )

    pointmap = F.interpolate(
        pointmap,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )  ## 1 x 3 x H x W

    return pointmap


def load_image_and_mask(input_dir, seg_dir, image_name):
    """Load BGR image and boolean foreground mask. Mask defaults to all-True if missing."""
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)

    mask = np.ones_like(image[:, :, 0], dtype=bool)
    if seg_dir is None:
        return image, mask

    mask_base = (
        image_name.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
    )
    npy_path = os.path.join(seg_dir, mask_base)
    candidates = [
        npy_path,
        npy_path.replace(".npy", "_seg.npy"),
        os.path.join(seg_dir, image_name),
    ]

    for mp in candidates:
        if not os.path.exists(mp):
            continue
        if mp.endswith("_seg.npy"):
            m = np.load(mp)  ## H x W, float; class labels
            mask = m > 0
        elif mp.endswith(".npy"):
            mask = np.load(mp)  ## H x W, boolean
        else:
            mask = cv2.imread(mp)[:, :, 0] > 0
        break

    return image, mask


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument(
        "--seg_dir", "--seg-dir", default=None, help="Path to segmentation dir"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--no-black-background",
        "--no_black_background",
        action="store_true",
        help="No black background",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="If provided, do not save .ply prediction files",
    )
    parser.add_argument(
        "--with-normal",
        "--with_normal",
        action="store_true",
        help="Also render surface-normal panel (default off — output is image | depth only).",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Get image list
    if os.path.isdir(args.input):
        input_dir = args.input
        image_names = [
            name
            for name in sorted(os.listdir(input_dir))
            if name.endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]
        input_dir = os.path.dirname(image_paths[0])

    seg_dir = args.seg_dir

    # Defer model init until we actually need to run inference (skip-if-cached).
    model = None

    # ============== Pass 1: inference (cached) + percentile collection ==============
    per_frame_percentiles = []  # list of (p1, p99) of foreground depth per frame
    for image_name in tqdm(image_names, desc="pass 1: inference"):
        base_path = os.path.join(args.output, image_name.rsplit(".")[0])
        depth_npy_path = f"{base_path}_depth.npy"

        image, mask = load_image_and_mask(input_dir, seg_dir, image_name)

        if os.path.exists(depth_npy_path):
            depth = np.load(depth_npy_path).astype(np.float32)
        else:
            if model is None:
                model = init_model(args.config, args.checkpoint, device=args.device)

            data = model.pipeline(dict(img=image))  ## resize and pad
            data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
            inputs, data_samples = data["inputs"], data["data_samples"]

            ## pointmap is 1 x 3 x H x W, scale is 1 x 1
            with torch.no_grad():
                pointmap, scale = model(inputs)
                pointmap = pointmap / scale  ## convert pointmap to metric

            assert (
                pointmap.shape[0] == 1
                and pointmap.shape[2] == inputs.shape[2]
                and pointmap.shape[3] == inputs.shape[3]
            )

            pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"][
                "padding_size"
            ]
            pointmap = pointmap[
                :,
                :,
                pad_top : inputs.shape[2] - pad_bottom,
                pad_left : inputs.shape[3] - pad_right,
            ]

            pointmap = resize_pointmap(
                pointmap,
                target_height=mask.shape[0],
                target_width=mask.shape[1],
                smooth=True,
            )
            pointmap = pointmap.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3

            depth = pointmap[:, :, 2].astype(np.float32)
            np.save(depth_npy_path, depth.astype(np.float16))

            if not args.no_save_predictions:
                points = pointmap[mask > 0].reshape(-1, 3)  ## N x 3
                pc = o3d.geometry.PointCloud()
                colors = image[mask > 0] / 255.0
                colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB

                pc.points = o3d.utility.Vector3dVector(points)
                pc.colors = o3d.utility.Vector3dVector(colors)
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.05, resolution=20
                )
                sphere.translate([0, 0, 0])  # Center the sphere at the origin
                sphere.paint_uniform_color([0, 0, 1])  # Color the sphere blue
                sphere_pc = sphere.sample_points_poisson_disk(number_of_points=500)
                sphere_pc.colors = o3d.utility.Vector3dVector(
                    [[0, 0, 1] for _ in range(len(sphere_pc.points))]
                )
                pc = pc + sphere_pc
                o3d.io.write_point_cloud(f"{base_path}.ply", pc)

        depth_fg = depth[mask > 0]
        if len(depth_fg) > 0:
            p1, p99 = np.percentile(depth_fg, [1, 99])
            per_frame_percentiles.append((float(p1), float(p99)))

    # ============== Aggregate global bounds: min(p1) and max(p99) ==============
    if per_frame_percentiles:
        arr = np.array(per_frame_percentiles)
        global_min = float(arr[:, 0].min())
        global_max = float(arr[:, 1].max())
        print(
            f"global depth bounds across {len(per_frame_percentiles)} frames: "
            f"min={global_min:.3f}, max={global_max:.3f}"
        )
    else:
        global_min, global_max = None, None
        print("warning: no foreground found in any frame, skipping render pass")

    # ============== Pass 2: render with global bounds (no inference) ==============
    for image_name in tqdm(image_names, desc="pass 2: render"):
        base_path = os.path.join(args.output, image_name.rsplit(".")[0])
        depth_npy_path = f"{base_path}_depth.npy"

        if not os.path.exists(depth_npy_path):
            continue

        image, mask = load_image_and_mask(input_dir, seg_dir, image_name)
        if not np.any(mask):
            continue

        depth = np.load(depth_npy_path).astype(np.float32)

        processed_depth = process_depth_map_with_bounds(
            depth, mask, global_min, global_max
        )
        panels = [image, processed_depth]
        if args.with_normal:
            normal_vis = compute_surface_normals(depth, mask, global_min, global_max)
            panels.append(normal_vis)

        if args.no_black_background:
            for p in panels[1:]:
                p[mask == 0] = image[mask == 0]

        vis_image = np.concatenate(panels, axis=1)
        cv2.imwrite(f"{base_path}{os.path.splitext(image_name)[1]}", vis_image)


if __name__ == "__main__":
    main()
