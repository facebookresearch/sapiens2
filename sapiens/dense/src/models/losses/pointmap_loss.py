# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sapiens.registry import MODELS

from .utils import weight_reduce_loss


@MODELS.register_module()
class PointmapShiftInvariantL1Loss(nn.Module):
    """L1 loss that is invariant to global translation of the point cloud"""

    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        eps=-100,
        loss_name="loss_shift_invariant",
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        valid_mask=None,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        ), f"Invalid reduction: {reduction_override}"

        reduction = reduction_override if reduction_override else self.reduction

        if valid_mask is None:
            valid_mask = (target > self.eps).detach().float()
            valid_mask = valid_mask[:, 0, :, :].unsqueeze(1)  # B x 1 x H x W

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        # Compute centroids for each batch
        pred_centroid = (pred * valid_mask).sum(dim=(2, 3)) / (
            valid_mask.sum(dim=(2, 3)) + 1e-6
        )  # (B, 3)
        target_centroid = (target * valid_mask).sum(dim=(2, 3)) / (
            valid_mask.sum(dim=(2, 3)) + 1e-6
        )  # (B, 3)

        # Center both point clouds
        pred_centered = pred - pred_centroid.view(pred.shape[0], 3, 1, 1)
        target_centered = target - target_centroid.view(target.shape[0], 3, 1, 1)

        # Compute L1 loss on centered points
        loss = torch.abs(pred_centered - target_centered) * valid_mask

        # Apply reduction
        if reduction == "mean":
            loss = loss.sum() / (valid_mask.sum().clamp(min=1))
        elif reduction == "sum":
            loss = loss.sum()

        loss = (
            weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight
        )

        # Handle numerical instabilities
        loss = torch.nan_to_num(
            loss,
            nan=torch.tensor(0, dtype=pred.dtype, device=pred.device),
            posinf=torch.tensor(0, dtype=pred.dtype, device=pred.device),
            neginf=torch.tensor(0, dtype=pred.dtype, device=pred.device),
        )

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name


def visualize_normals(pred_normals, target_normals, valid_mask):
    # Set random seed
    seed = np.random.randint(0, 100000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Convert tensors to numpy arrays
    pred = pred_normals.detach().cpu().numpy()
    target = target_normals.detach().cpu().numpy()
    mask = valid_mask.detach().cpu().numpy()

    def process_normal_map(normal_map, mask):
        # Take the first sample in batch
        normal_map = normal_map[0]  # (3, H, W)
        mask = mask[0, 0]  # (H, W)

        # Print unique values for each channel
        print("\nNormal map statistics:")
        print("-" * 50)
        for i, axis in enumerate(["X", "Y", "Z"]):
            unique_vals = np.unique(normal_map[i])
            print(f"\n{axis}-component unique values:")
            print(f"Min: {normal_map[i].min():.4f}")
            print(f"Max: {normal_map[i].max():.4f}")
            print(f"Number of unique values: {len(unique_vals)}")
            print(
                f"Sample of unique values: {unique_vals[:10]}"
            )  # Show first 10 unique values

        # Print statistics for valid regions only
        valid_normals = normal_map[:, mask > 0]
        print("\nValid regions statistics:")
        print("-" * 50)
        for i, axis in enumerate(["X", "Y", "Z"]):
            if valid_normals.size > 0:
                print(f"\n{axis}-component (valid regions):")
                print(f"Min: {valid_normals[i].min():.4f}")
                print(f"Max: {valid_normals[i].max():.4f}")
                print(f"Mean: {valid_normals[i].mean():.4f}")
                print(f"Std: {valid_normals[i].std():.4f}")

        # Calculate and print vector magnitudes
        magnitudes = np.linalg.norm(normal_map, axis=0)
        print("\nNormal vector magnitudes:")
        print(f"Min magnitude: {magnitudes.min():.4f}")
        print(f"Max magnitude: {magnitudes.max():.4f}")
        print(f"Mean magnitude: {magnitudes.mean():.4f}")

        # Convert to RGB space [0, 255]
        # Map from [-1, 1] to [0, 1]
        normal_rgb = (normal_map + 1.0) * 0.5
        normal_rgb = np.clip(normal_rgb * 255, 0, 255).astype(np.uint8)

        # Transpose from (3, H, W) to (H, W, 3)
        normal_rgb = np.transpose(normal_rgb, (1, 2, 0))

        # Apply mask
        normal_rgb[mask == 0] = 0

        return normal_rgb

    # Process both prediction and ground truth
    pred_vis = process_normal_map(pred, mask)
    gt_vis = process_normal_map(target, mask)

    # Save images
    cv2.imwrite(f"seed_{seed}_pred.jpg", cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"seed_{seed}_gt.jpg", cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))

    return pred_vis, gt_vis


@MODELS.register_module()
class PointmapNormalLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        eps=-100,
        l1_weight=1.0,
        loss_name="loss_normal",
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.l1_weight = l1_weight  # Weight for L1 loss component
        self._loss_name = loss_name

    def compute_normals(self, pointmap, valid_mask, scale=1.0, amplification=100.0):
        """
        Compute surface normals from pointmap using neighboring points
        Args:
            pointmap: (B, 3, H, W) tensor of XYZ coordinates
            valid_mask: (B, 1, H, W) binary mask
        Returns:
            normals: (B, 3, H, W) tensor of normalized surface normals
        """
        ## scale canonical Z to metric Z
        scaled_pointmap = pointmap.clone()
        scale = scale.view(-1, 1, 1, 1)
        scaled_pointmap = (
            pointmap / scale
        )  ## scale canonical pointmap to metric pointmap

        # Pad pointmap for computing neighbors
        pointmap_pad = F.pad(scaled_pointmap, (1, 1, 1, 1), mode="reflect")
        center = pointmap_pad[:, :, 1:-1, 1:-1]

        # Get vectors to neighbors (use central differences)
        dx = pointmap_pad[:, :, 1:-1, 2:] - center  # right - center
        dy = pointmap_pad[:, :, 2:, 1:-1] - center  # down - center

        dx = dx * amplification  # Scale up x gradient
        dy = dy * amplification  # Scale up y gradient

        # Compute cross product for normal vectors
        normal = torch.cross(dx, dy, dim=1)  # (B, 3, H, W)

        # Normalize the vectors
        normal_magnitude = torch.norm(normal, dim=1, keepdim=True)
        normal = normal / normal_magnitude.clamp(min=1e-6)
        # normal = torch.where(
        # mask, normal / (normal_magnitude + self.thres_eps), torch.zeros_like(normal)
        # )

        # Apply valid mask and handle invalid normals
        magnitude_is_valid = normal_magnitude > 1e-6
        valid_normals = magnitude_is_valid & (valid_mask > 0)
        normal = normal * valid_normals.float()

        return normal, valid_normals

    def forward(
        self,
        pred,
        target,
        valid_mask=None,
        scale=None,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """
        Args:
            pred (Tensor): Predicted pointmap of shape (B, 3, H, W)
            target (Tensor): Target pointmap of shape (B, 3, H, W)
            valid_mask (Tensor, optional): Validity mask of shape (B, 1, H, W)
        """
        assert pred.shape == target.shape
        assert reduction_override in (None, "none", "mean", "sum")

        reduction = reduction_override if reduction_override else self.reduction

        ## metric Z = canonical Z / scale
        if valid_mask is None:
            valid_mask = (target > self.eps).detach().float()
            valid_mask = valid_mask[:, 0, :, :].unsqueeze(1)

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        # Compute normals with validity masks
        pred_normals, _ = self.compute_normals(pred, valid_mask, scale)
        target_normals, target_valid = self.compute_normals(target, valid_mask, scale)
        # Combined validity mask for normal comparison
        valid = target_valid  ## only use gt pointmap

        # Cosine similarity loss (1 - cos)
        cos_similarity = torch.sum(pred_normals * target_normals, dim=1, keepdim=True)
        cos_similarity = torch.clamp(cos_similarity, min=-1.0, max=1.0)
        normal_loss = (1 - cos_similarity) * valid.float()

        # L1 loss on normal vectors
        l1_loss = (
            torch.abs(pred_normals - target_normals).mean(dim=1, keepdim=True)
            * valid.float()
        )

        # Combine losses
        combined_loss = normal_loss + self.l1_weight * l1_loss

        # Apply reduction
        if reduction == "mean":
            loss = combined_loss.sum() / (valid.sum().clamp(min=1))
        elif reduction == "sum":
            loss = combined_loss.sum()
        else:
            loss = combined_loss

        loss = (
            weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight
        )

        # ----------debug-----------
        # visualize_normals(pred_normals, target_normals, valid)

        return loss

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class PointmapIntrinsicsConsistencyLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        eps=-100,
        loss_name="loss_K_consistency",
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        valid_mask=None,
        intrinsics=None,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert pred.shape == target.shape, (
            f"The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched"
        )
        assert reduction_override in (
            None,
            "none",
            "mean",
            "sum",
        ), "Invalid reduction_override value"
        assert intrinsics is not None, (
            "intrinsics must be provided for PointmapIntrinsicsConsistencyLoss"
        )
        assert intrinsics.shape[1] == 3 and intrinsics.shape[2] == 3, (
            "intrinsics must be a B x 3 x 3 tensor"
        )

        reduction = reduction_override if reduction_override else self.reduction

        if valid_mask is None:
            valid_mask = (target > self.eps).detach().float()
            valid_mask = valid_mask[:, 0, :, :].unsqueeze(1)  ## B x 1 x H x W

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        B, C, H, W = pred.shape
        device = pred.device
        valid_mask = valid_mask.squeeze(1)  ## B x H x W

        pred_X = pred[:, 0, :, :]  ## B x H x W
        pred_Y = pred[:, 1, :, :]  ## B x H x W
        pred_Z = pred[:, 2, :, :]  ## B x H x W

        cols = torch.arange(W, device=device).repeat(B, H, 1)  # B x H x W
        rows = (
            torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)
        )  # B x H x W

        # Compute x and y from z and K
        x = (
            (cols - intrinsics[:, 0, 2].view(B, 1, 1))
            * pred_Z
            / intrinsics[:, 0, 0].view(B, 1, 1)
        )  # B x H x W
        y = (
            (rows - intrinsics[:, 1, 2].view(B, 1, 1))
            * pred_Z
            / intrinsics[:, 1, 1].view(B, 1, 1)
        )  # B x H x W

        # # ##---------------to debug consistency with target-------------------
        # target_X = target[:, 0, :, :]
        # target_Y = target[:, 1, :, :]
        # x = (cols - intrinsics[:, 0, 2].view(B, 1, 1)) * target_Z / intrinsics[:, 0, 0].view(B, 1, 1)
        # y = (rows - intrinsics[:, 1, 2].view(B, 1, 1)) * target_Z / intrinsics[:, 1, 1].view(B, 1, 1)

        # loss_X = torch.abs(target_X - x) * valid_mask
        # loss_Y = torch.abs(target_Y - y) * valid_mask

        ##-----------------------------------------------------
        # Loss calculations
        loss_X = torch.abs(pred_X - x) * valid_mask
        loss_Y = torch.abs(pred_Y - y) * valid_mask

        # Apply reduction
        if self.reduction == "mean":
            loss = (loss_X + loss_Y).sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            loss = (loss_X + loss_Y).sum()
        elif self.reduction == "none":
            loss = loss_X + loss_Y
        else:
            raise ValueError("Unsupported reduction type")

        # Handle NaN values
        loss = torch.nan_to_num(loss, nan=0.0) * self.loss_weight

        return loss

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class PointmapScaleL1Loss(nn.Module):
    """L1 loss that is invariant to global translation of the point cloud"""

    def __init__(self, loss_weight=1.0, loss_name="loss_scale_l1", **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target, reduction="mean") * self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
