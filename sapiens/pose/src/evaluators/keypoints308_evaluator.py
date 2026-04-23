# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sapiens.engine.evaluators import BaseEvaluator
from sapiens.registry import MODELS

from ..datasets.codecs import UDPHeatmap
from ..datasets.codecs.utils import get_heatmap_maximum
from ..datasets.utils import parse_pose_metainfo

## get the keypoints ids
try:
    this_file = os.path.abspath(__file__)
    root_dir = os.path.abspath(os.path.join(this_file, "..", "..", ".."))
    sys.path.append(str(os.path.join(root_dir)))
    from configs._base_.keypoints308 import dataset_info as KEYPOINTS308_INFO

    KEYPOINTS308_INFO["name2id"] = {}
    for keypoint_id, keypoint_info in KEYPOINTS308_INFO["keypoint_info"].items():
        KEYPOINTS308_INFO["name2id"][keypoint_info["name"]] = keypoint_id

    KEYPOINTS308_INFO["body_keypoint_ids"] = [
        KEYPOINTS308_INFO["name2id"][name]
        for name in KEYPOINTS308_INFO["body_keypoint_names"]
    ]

    KEYPOINTS308_INFO["foot_keypoint_ids"] = [
        KEYPOINTS308_INFO["name2id"][name]
        for name in KEYPOINTS308_INFO["foot_keypoint_names"]
    ]

    KEYPOINTS308_INFO["face_keypoint_ids"] = [
        KEYPOINTS308_INFO["name2id"][name]
        for name in KEYPOINTS308_INFO["face_keypoint_names"]
    ]

    KEYPOINTS308_INFO["left_hand_keypoint_ids"] = [
        KEYPOINTS308_INFO["name2id"][name]
        for name in KEYPOINTS308_INFO["left_hand_keypoint_names"]
    ]

    KEYPOINTS308_INFO["right_hand_keypoint_ids"] = [
        KEYPOINTS308_INFO["name2id"][name]
        for name in KEYPOINTS308_INFO["right_hand_keypoint_names"]
    ]

except Exception as e:
    pass


@MODELS.register_module()
class Keypoints308Evaluator(BaseEvaluator):
    body_num = 17
    foot_num = 6
    face_num = 238
    left_hand_num = 20
    right_hand_num = 20
    remaining_extra_num = 7  ## total to 308

    def __init__(
        self,
        decoder: Optional[dict] = None,
        ann_file: Optional[str] = None,
        use_area: bool = True,
        iou_type: str = "keypoints",
        score_mode: str = "bbox_keypoint",
        keypoint_score_thr: float = 0.2,
        nms_mode: str = "oks_nms",
        nms_thr: float = 0.9,
    ):
        from xtcocotools.coco import COCO  # lazy: only needed for COCO-format ann files

        super().__init__()
        self.num_keypoints = 308
        decoder_type = decoder.pop("type")
        assert decoder_type == "UDPHeatmap", "Only UDPHeatmap is supported"
        self.decoder = UDPHeatmap(**decoder)
        self.coco = COCO(ann_file)

        self.dataset_meta = parse_pose_metainfo(
            dict(from_file="configs/_base_/keypoints308.py")
        )

        self.body_keypoint_ids = KEYPOINTS308_INFO["body_keypoint_ids"]
        self.foot_keypoint_ids = KEYPOINTS308_INFO["foot_keypoint_ids"]
        self.face_keypoint_ids = KEYPOINTS308_INFO["face_keypoint_ids"]
        self.left_hand_keypoint_ids = KEYPOINTS308_INFO["left_hand_keypoint_ids"]
        self.right_hand_keypoint_ids = KEYPOINTS308_INFO["right_hand_keypoint_ids"]

        assert len(self.body_keypoint_ids) == self.body_num
        assert len(self.foot_keypoint_ids) == self.foot_num
        assert len(self.face_keypoint_ids) == self.face_num
        assert len(self.left_hand_keypoint_ids) == self.left_hand_num
        assert len(self.right_hand_keypoint_ids) == self.right_hand_num

        self.use_area = use_area
        self.iou_type = iou_type

        allowed_score_modes = ["bbox", "bbox_keypoint", "bbox_rle", "keypoint"]
        if score_mode not in allowed_score_modes:
            raise ValueError(
                "`score_mode` should be one of 'bbox', 'bbox_keypoint', "
                f"'bbox_rle', but got {score_mode}"
            )
        self.score_mode = score_mode
        self.keypoint_score_thr = keypoint_score_thr

        allowed_nms_modes = ["oks_nms"]
        if nms_mode not in allowed_nms_modes:
            raise ValueError(
                "`nms_mode` should be one of 'oks_nms', but got {nms_mode}"
            )
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr

    @torch.no_grad()
    def process(self, predictions: torch.Tensor, data_samples: dict, accelerator=None):
        assert accelerator is not None, "evaluation process expects an accelerator"

        if predictions.dtype == torch.bfloat16:
            predictions = predictions.float()

        pred_heatmaps = predictions.cpu().numpy()  ## B x K x heatmap_H x heatmap_W
        (
            keypoints_list,
            scores_list,
            ids_list,
            img_ids_list,
            areas_list,
            bbox_scores_list,
        ) = ([], [], [], [], [], [])

        for i in range(pred_heatmaps.shape[0]):
            pred_heatmap = pred_heatmaps[i]
            meta_sample = data_samples[i]["meta"]  # Assuming 'meta' is a list of dicts

            keypoints, keypoint_scores = self.decoder.decode(
                pred_heatmap
            )  ## kps in crop image

            ## convert to global image size
            bbox_center = meta_sample["bbox_center"]  ## 1 x 2
            bbox_scale = meta_sample["bbox_scale"]  ## 1 x 2
            input_size = np.array(meta_sample["input_size"])  ## 2, 768 x 1024
            area = np.prod(meta_sample["bbox_scale"])

            keypoints = (
                keypoints / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
            )
            keypoints_list.append(keypoints)
            scores_list.append(keypoint_scores)
            ids_list.append(int(meta_sample["id"]))
            img_ids_list.append(int(meta_sample["img_id"]))
            bbox_scores_list.append(meta_sample["bbox_score"])
            areas_list.append(area)

        if not areas_list:
            areas_list = [0.0] * len(keypoints_list)

        results_to_gather = {
            "keypoints": torch.tensor(
                np.array(keypoints_list), device=predictions.device
            ),
            "keypoint_scores": torch.tensor(
                np.array(scores_list), device=predictions.device
            ),
            "id": torch.tensor(ids_list, device=predictions.device),
            "img_id": torch.tensor(img_ids_list, device=predictions.device),
            "areas": torch.tensor(areas_list, device=predictions.device),
            "bbox_scores": torch.tensor(
                np.array(bbox_scores_list), device=predictions.device
            ),
        }
        gathered_results = accelerator.gather_for_metrics(results_to_gather)

        if accelerator.is_main_process:
            keypoints_all = gathered_results["keypoints"].cpu().numpy()
            scores_all = gathered_results["keypoint_scores"].cpu().numpy()
            ids_all = gathered_results["id"].cpu().tolist()
            img_ids_all = gathered_results["img_id"].cpu().tolist()
            areas_all = gathered_results["areas"].cpu().tolist()
            bbox_scores_all = gathered_results["bbox_scores"].cpu().tolist()

            for i in range(len(keypoints_all)):
                pred = {
                    "id": ids_all[i],
                    "img_id": img_ids_all[i],
                    "keypoints": keypoints_all[i],
                    "keypoint_scores": scores_all[i],
                    "areas": areas_all[i],
                    "category_id": 1,  # Defaulting category_id
                    "bbox_scores": bbox_scores_all[i],
                }
                # Assuming self.results is the master list for the evaluator
                self.results.append(pred)
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

        kpts = defaultdict(list)

        print("len of results: ", len(self.results))
        for pred in self.results:
            img_id = pred["img_id"]
            for idx in range(len(pred["keypoints"])):
                instance = {
                    "id": pred["id"],
                    "img_id": pred["img_id"],
                    "category_id": pred["category_id"],
                    "keypoints": pred["keypoints"][idx],  ## K x 2
                    "keypoint_scores": pred["keypoint_scores"][idx],  ## K
                    "bbox_score": pred["bbox_scores"][idx],
                }

                # use keypoint to calculate bbox and get area
                keypoints = pred["keypoints"][idx]
                area = (np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                    np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
                )
                instance["area"] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key="id")
        valid_kpts = defaultdict(list)
        num_keypoints = self.num_keypoints

        assert len(self.dataset_meta["sigmas"]) == num_keypoints

        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance["keypoints"] = np.concatenate(
                    [instance["keypoints"], instance["keypoint_scores"][:, None]],
                    axis=-1,
                )
                if self.score_mode == "bbox_keypoint":
                    bbox_score = instance["bbox_score"]
                    mean_kpt_score = 0
                    valid_num = 0
                    for kpt_idx in range(num_keypoints):
                        kpt_score = instance["keypoint_scores"][kpt_idx]
                        if kpt_score > self.keypoint_score_thr:
                            mean_kpt_score += kpt_score
                            valid_num += 1
                    if valid_num != 0:
                        mean_kpt_score /= valid_num
                    instance["score"] = bbox_score * mean_kpt_score
            # perform nms
            nms = oks_nms if self.nms_mode == "oks_nms" else None
            keep = nms(instances, self.nms_thr, sigmas=self.dataset_meta["sigmas"])
            valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = os.path.join(tmp_dir.name, "results")
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f"Evaluating {self.__class__.__name__}...")
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if logger is not None:
            logger.info(info_str)

        self.reset()

        return eval_results

    def results2json(self, keypoints: Dict[int, list], outfile_prefix: str = "") -> str:
        # the results with category_id
        cat_id = 1
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array([img_kpt["keypoints"] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta["num_keypoints"]

            _body_keypoints = _keypoints[
                :, self.body_keypoint_ids
            ].copy()  ## get only body keypoints
            _foot_keypoints = _keypoints[
                :, self.foot_keypoint_ids
            ].copy()  ## get only foot keypoints
            _face_keypoints = _keypoints[
                :, self.face_keypoint_ids
            ].copy()  ## get only face keypoints
            _left_hand_keypoints = _keypoints[
                :, self.left_hand_keypoint_ids
            ].copy()  ## get only left hand keypoints
            _right_hand_keypoints = _keypoints[
                :, self.right_hand_keypoint_ids
            ].copy()  ## get only right hand keypoints

            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)  ## flatten
            _body_keypoints = _body_keypoints.reshape(-1, self.body_num * 3)  ## flatten
            _foot_keypoints = _foot_keypoints.reshape(-1, self.foot_num * 3)  ## flatten
            _face_keypoints = _face_keypoints.reshape(-1, self.face_num * 3)  ## flatten
            _left_hand_keypoints = _left_hand_keypoints.reshape(
                -1, self.left_hand_num * 3
            )  ## flatten
            _right_hand_keypoints = _right_hand_keypoints.reshape(
                -1, self.right_hand_num * 3
            )  ## flatten

            result = [
                {
                    "image_id": img_kpt["img_id"],
                    "category_id": cat_id,
                    "goliath_wholebody_kpts": _keypoint.tolist(),  ## all keypoints. Modified in xtcocotools
                    "keypoints": _body_keypoint.tolist(),  ## xtcocotools treats this as body keypoints, 17 default
                    "foot_kpts": _foot_keypoint.tolist(),
                    "face_kpts": _face_keypoint.tolist(),
                    "lefthand_kpts": _left_hand_keypoint.tolist(),
                    "righthand_kpts": _right_hand_keypoint.tolist(),
                    "score": float(img_kpt["score"]),
                }
                for img_kpt, _keypoint, _body_keypoint, _foot_keypoint, _face_keypoint, _left_hand_keypoint, _right_hand_keypoint in zip(
                    img_kpts,
                    _keypoints,
                    _body_keypoints,
                    _foot_keypoints,
                    _face_keypoints,
                    _left_hand_keypoints,
                    _right_hand_keypoints,
                )
            ]

            cat_results.extend(result)

        res_file = f"{outfile_prefix}.keypoints.json"
        json.dump(cat_results, open(res_file, "w"), sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        from xtcocotools.cocoeval import COCOeval  # lazy: only needed during eval

        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta["sigmas"]

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            "keypoints_body",
            sigmas[self.body_keypoint_ids],
            use_area=True,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            "keypoints_foot",
            sigmas[self.foot_keypoint_ids],
            use_area=True,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            "keypoints_face",
            sigmas[self.face_keypoint_ids],
            use_area=True,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            "keypoints_lefthand",
            sigmas[self.left_hand_keypoint_ids],
            use_area=True,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            "keypoints_righthand",
            sigmas[self.right_hand_keypoint_ids],
            use_area=True,
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco, coco_det, "keypoints_wholebody_goliath", sigmas, use_area=True
        )
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            "AP",
            "AP .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(
        self, kpts: Dict[int, list], key: str = "id"
    ) -> Dict[int, list]:
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts


# -------------------------------------------------------------------------------
def nms(dets: np.ndarray, thr: float) -> List[int]:
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(
    g: np.ndarray,
    d: np.ndarray,
    a_g: float,
    a_d: np.ndarray,
    sigmas: Optional[np.ndarray] = None,
    vis_thr: Optional[float] = None,
) -> np.ndarray:
    if sigmas is None:
        sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list((vg > vis_thr) & (vd > vis_thr))
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(
    kpts_db: List[dict],
    thr: float,
    sigmas: Optional[np.ndarray] = None,
    vis_thr: Optional[float] = None,
    score_per_joint: bool = False,
):
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k["score"].mean() for k in kpts_db])
    else:
        scores = np.array([k["score"] for k in kpts_db])

    kpts = np.array([k["keypoints"].flatten() for k in kpts_db])
    areas = np.array([k["area"] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(
            kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, vis_thr
        )

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _calc_distances(
    preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray
) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1
    )
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    thr: np.ndarray,
    norm_factor: np.ndarray,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def pose_pck_accuracy(
    output: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    thr: float = 0.05,
    normalize: Optional[np.ndarray] = None,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = get_heatmap_maximum(output)
    gt, _ = get_heatmap_maximum(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)
