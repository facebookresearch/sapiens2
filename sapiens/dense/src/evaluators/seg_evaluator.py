# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from sapiens.engine.evaluators import BaseEvaluator
from sapiens.registry import MODELS

from ..datasets.seg.seg_dome_dataset import DOME_CLASSES_29


@MODELS.register_module()
class SegEvaluator(BaseEvaluator):
    def __init__(
        self,
        class_names="dome29",
        ignore_index: int = 255,
        iou_metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_names = (
            self.extract_class(DOME_CLASSES_29) if class_names == "dome29" else None
        )
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta

    def extract_class(self, class_names):
        return [class_info["name"] for _, class_info in class_names.items()]

    @torch.no_grad()
    def process(self, pred_logits, data_samples: dict, accelerator=None):
        assert accelerator is not None, "evaluation process expects an accelerator"
        num_classes = pred_logits.shape[1]

        ai_list, au_list, apl_list, al_list = [], [], [], []
        for i in range(len(pred_logits)):
            pred_logit = pred_logits[i]  # C x H x W
            gt_label = data_samples[i]["gt_seg"].squeeze()  # H x W

            if pred_logit.shape[2:] != gt_label.shape:
                pred_logit = F.interpolate(
                    input=pred_logit.unsqueeze(0),
                    size=gt_label.shape,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                ).squeeze(0)

            pred_label = pred_logit.argmax(dim=0)  # H x W

            a_i, a_u, a_pl, a_l = self.intersect_and_union(
                pred_label, gt_label, num_classes, self.ignore_index
            )

            ai_list.append(a_i)
            au_list.append(a_u)
            apl_list.append(a_pl)
            al_list.append(a_l)

        # Local per-batch tensors: (B_local, C)
        ai = torch.stack(ai_list, dim=0)
        au = torch.stack(au_list, dim=0)
        apl = torch.stack(apl_list, dim=0)
        al = torch.stack(al_list, dim=0)

        # Pack as (B_local, 4, C) so gather concatenates along the batch dim.
        pack = torch.stack([ai, au, apl, al], dim=1)  # (B_local, 4, C)
        gpack = accelerator.gather_for_metrics(pack)  # (B_global_this_step, 4, C)
        batch_tot = gpack.sum(dim=0)  # (4, C) global for this step

        ai_g, au_g, apl_g, al_g = batch_tot[0], batch_tot[1], batch_tot[2], batch_tot[3]

        # Only rank-0 appends real totals for this batch
        if accelerator.is_main_process:
            self.results.append((ai_g, au_g, apl_g, al_g))

        return

    def evaluate(self, logger=None, accelerator=None) -> Dict[str, float]:
        assert accelerator is not None, "evaluation aggregation expects an accelerator"

        if not accelerator.is_main_process:
            self.reset()
            return {}

        if not self.results:
            if logger is not None:
                logger.info("No results to evaluate.")
            return {}

        per_field = list(zip(*self.results))  # [(ai_b), (au_b), (apl_b), (al_b)]
        totals = [torch.stack(x, dim=0).sum(dim=0) for x in per_field]

        (
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
        ) = totals  # tensors already reduced across ranks

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        if self.class_names is not None:
            ret_metrics_class.update({"Class": self.class_names})
            ret_metrics_class.move_to_end("Class", last=False)
            class_table_data = PrettyTable()
            for key, val in ret_metrics_class.items():
                class_table_data.add_column(key, val)

            logger.info("\n" + class_table_data.get_string())

        self.reset()
        return metrics

    def intersect_and_union(
        self,
        pred_label: torch.tensor,
        label: torch.tensor,
        num_classes: int,
        ignore_index: int,
    ):
        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    def total_area_to_metrics(
        self,
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        def f_score(precision, recall, beta=1):
            score = (
                (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f"metrics {metrics} is not supported")

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = {
            metric: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
            )
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics
