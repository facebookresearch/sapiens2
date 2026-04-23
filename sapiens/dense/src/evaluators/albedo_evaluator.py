# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from sapiens.engine.evaluators import BaseEvaluator
from sapiens.registry import MODELS


@MODELS.register_module()
class AlbedoEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self._psnr_data_range: float | None = None  # set on first batch

    @staticmethod
    def _gaussian_kernel(ks: int = 11, sigma: float = 1.5, device=None, dtype=None):
        ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        k = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        k = k / k.sum()
        return k

    @torch.no_grad()
    def _masked_ssim_sum(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        data_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        pred, gt: (3, H, W), mask: (H, W) bool/0-1
        Returns (sum_ssim, count_ssim) across valid windows.
        """
        eps = 1e-8
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        if pred.dtype == torch.bfloat16:
            pred = pred.float()
            gt = gt.float()
            mask = mask.float()

        x = pred.unsqueeze(0)  # (1,3,H,W)
        y = gt.unsqueeze(0)  # (1,3,H,W)
        m = mask.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)  # (1,1,H,W)

        B, C, H, W = x.shape
        k = self._gaussian_kernel(ks=11, sigma=1.5, device=x.device, dtype=x.dtype)
        pad = 11 // 2
        k_img = k.view(1, 1, 11, 11)
        k_ch = k_img.repeat(C, 1, 1, 1)  # grouped conv kernel

        # local normalization with mask
        m_conv = F.conv2d(m, k_img, padding=pad)  # (1,1,H,W)
        m_conv = torch.clamp(m_conv, min=eps)

        def _conv(z):
            return F.conv2d(z, k_ch, padding=pad, groups=C)

        x_m = x * m
        y_m = y * m

        mu_x = _conv(x_m) / m_conv
        mu_y = _conv(y_m) / m_conv

        x2_m = (x * x) * m
        y2_m = (y * y) * m
        xy_m = (x * y) * m

        sigma_x2 = _conv(x2_m) / m_conv - mu_x * mu_x
        sigma_y2 = _conv(y2_m) / m_conv - mu_y * mu_y
        sigma_xy = _conv(xy_m) / m_conv - mu_x * mu_y

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map_ch = num / (den + eps)  # (1,C,H,W)
        ssim_map = ssim_map_ch.mean(
            dim=1, keepdim=True
        )  # average over channels -> (1,1,H,W)

        # Only count windows with sufficient valid support
        valid_win = (m_conv > 0.5).squeeze(0).squeeze(0)  # (H,W)
        sum_ssim = ssim_map.squeeze(0).squeeze(0)[valid_win].to(torch.float64).sum()
        cnt_ssim = valid_win.to(torch.float64).sum()
        return sum_ssim, cnt_ssim

    @torch.no_grad()
    def process(self, predictions: torch.Tensor, data_samples: dict, accelerator=None):
        """
        Args:
            predictions: Tensor, predicted albedo (B, 3, H_low, W_low)
            data_samples: dict with keys:
                - "mask": (B, 1, H, W) >0 is valid
                - "gt_albedo": (B, 3, H, W)
        """
        assert accelerator is not None, "evaluation process expects an accelerator"
        pred_albedos = predictions  # (B,3,h,w)
        gt_masks = data_samples["mask"]  # (B,1,H,W)
        gt_albedos = data_samples["gt_albedo"]  # (B,3,H,W)

        # align spatial size
        if pred_albedos.shape[2:] != gt_albedos.shape[2:]:
            pred_albedos = F.interpolate(
                input=pred_albedos,
                size=gt_albedos.shape[2:],
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

        # set PSNR range (once)
        if self._psnr_data_range is None:
            mx = gt_albedos.detach().max()
            self._psnr_data_range = 255.0 if mx > 1.5 else 1.0

        B = gt_albedos.shape[0]
        per_sample_vecs = []  # each: [sum_l1, sum_l2, N_pix, sum_grad_l1, N_grad, sum_ssim, N_ssim]

        for i in range(B):
            mask = gt_masks[i, 0] > 0
            n_valid = int(mask.sum().item())
            assert n_valid > 0, "no valid pixels found"

            gt = gt_albedos[i]  # (3,H,W)
            pr = pred_albedos[i]  # (3,H,W)

            # --- Pixel MAE / RMSE accumulators (average over channels per pixel) ---
            diff = pr - gt
            l1_pix = diff.abs().mean(dim=0)  # (H,W)
            l2_pix = (diff * diff).mean(dim=0)  # (H,W)

            sum_l1 = l1_pix[mask].to(torch.float64).sum().unsqueeze(0)  # (1,)
            sum_l2 = l2_pix[mask].to(torch.float64).sum().unsqueeze(0)  # (1,)
            N_pix = torch.tensor(
                [float(n_valid)], dtype=torch.float64, device=pr.device
            )

            # --- Gradient L1 (simple forward differences; mask both sides) ---
            # horizontal
            mask_h = mask[:, 1:] & mask[:, :-1]
            dx_pr = pr[:, :, 1:] - pr[:, :, :-1]
            dx_gt = gt[:, :, 1:] - gt[:, :, :-1]
            grad_l1_h = (dx_pr - dx_gt).abs().mean(dim=0)  # (H,W-1)
            sum_grad_h = grad_l1_h[mask_h].to(torch.float64).sum()
            N_grad_h = mask_h.to(torch.float64).sum()

            # vertical
            mask_v = mask[1:, :] & mask[:-1, :]
            dy_pr = pr[:, 1:, :] - pr[:, :-1, :]
            dy_gt = gt[:, 1:, :] - gt[:, :-1, :]
            grad_l1_v = (dy_pr - dy_gt).abs().mean(dim=0)  # (H-1,W)
            sum_grad_v = grad_l1_v[mask_v].to(torch.float64).sum()
            N_grad_v = mask_v.to(torch.float64).sum()

            sum_grad_l1 = (sum_grad_h + sum_grad_v).unsqueeze(0)  # (1,)
            N_grad = (N_grad_h + N_grad_v).unsqueeze(0)  # (1,)

            # --- SSIM (masked, with Gaussian window) ---
            sum_ssim, cnt_ssim = self._masked_ssim_sum(
                pr, gt, mask, data_range=float(self._psnr_data_range)
            )
            sum_ssim = sum_ssim.unsqueeze(0)  # (1,)
            N_ssim = cnt_ssim.unsqueeze(0)  # (1,)

            vec = torch.cat(
                [sum_l1, sum_l2, N_pix, sum_grad_l1, N_grad, sum_ssim, N_ssim], dim=0
            )
            per_sample_vecs.append(vec)

        pack = torch.stack(per_sample_vecs, dim=0)  # (B_local, 7)
        gpack = accelerator.gather_for_metrics(pack)  # (B_global_step, 7)
        step_totals = gpack.sum(dim=0)  # (7,)

        if accelerator.is_main_process:
            self.results.append(step_totals)
        return

    def evaluate(self, logger=None, accelerator=None) -> Dict[str, float]:
        """
        Returns:
            Dict[str, float]: {
                'albedo_mae', 'albedo_rmse', 'albedo_psnr',
                'albedo_ssim', 'albedo_grad_l1'
            }
        """
        assert accelerator is not None, "evaluation aggregation expects an accelerator"

        if not accelerator.is_main_process:
            self.reset()
            return {}

        if not self.results:
            if logger is not None:
                logger.info("No results to evaluate.")
            return {}

        totals = torch.stack(self.results, dim=0).sum(dim=0)  # (7,)
        idx = 0
        sum_l1 = totals[idx]
        idx += 1
        sum_l2 = totals[idx]
        idx += 1
        N_pix = totals[idx]
        idx += 1
        sum_grad_l1 = totals[idx]
        idx += 1
        N_grad = totals[idx]
        idx += 1
        sum_ssim = totals[idx]
        idx += 1
        N_ssim = totals[idx]
        idx += 1

        # Core metrics
        mae = (sum_l1 / N_pix).item()
        mse = (sum_l2 / N_pix).clamp_min(1e-12)
        rmse = torch.sqrt(mse).item()

        # PSNR
        L2 = float(self._psnr_data_range or 1.0) ** 2
        psnr = (10.0 * torch.log10(torch.tensor(L2, dtype=torch.float64) / mse)).item()

        # SSIM (mean over valid windows)
        ssim = (sum_ssim / torch.clamp_min(N_ssim, 1.0)).item()

        # Gradient L1
        grad_l1 = (sum_grad_l1 / torch.clamp_min(N_grad, 1.0)).item()

        metrics: Dict[str, float] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "grad_l1": float(grad_l1),
        }

        table = PrettyTable()
        table.field_names = list(metrics.keys())
        table.add_row([f"{float(v):.5f}" for v in metrics.values()])
        if logger is not None:
            logger.info("\n" + table.get_string())

        self.reset()
        return metrics
