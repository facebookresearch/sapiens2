# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from argparse import ArgumentParser

# Block mmpretrain: mmdet's reid modules try `import mmpretrain` inside
# try/except ImportError, but mmpretrain's BLIP language_model.py raises
# TypeError (transformers API drift) — escapes the except and kills the process.
# We don't use reid or mmpretrain, so force a clean ImportError.
sys.modules["mmpretrain"] = None

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.evaluators import nms
from sapiens.pose.models import init_model
from tqdm import tqdm

from pose_render_utils import visualize_keypoints

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def mmdet_pipeline(cfg):
    from mmdet.datasets import transforms

    if "test_dataloader" not in cfg:
        return cfg
    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans["type"] in dir(transforms):
            trans["type"] = "mmdet." + trans["type"]
    return cfg


def process_one_image(args, image, detector, model):
    image_w, image_h = image.shape[1], image.shape[0]
    det_result = inference_detector(detector, image)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == 0,  ## 0 is the person class
            pred_instance.scores > args.bbox_thr,
        )
    ]

    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]  ## B x 4; x1, y1, x2, y2
    # get bbox from the image size
    if bboxes is None or len(bboxes) == 0:
        bboxes = np.array([[0, 0, image_w - 1, image_h - 1]], dtype=np.float32)

    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)  # B x 3 x H x W
    with torch.no_grad():
        pred = model(inputs)  # B x 3 x H x W
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))  # B x 3 x H x W
            pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]  ## K
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0

    # ------------------------------------------
    pred = pred.cpu().numpy()  ## B x K x heatmap_H x heatmap_W
    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        ## kps in crop image
        ## keypoints_i is 1 x K x 2
        # keypoint_scores_i is 1 x K
        keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
        input_size = data_samples["meta"]["input_size"]  ## 1 x 2, 768 x 1024
        bbox_center = data_samples["meta"]["bbox_center"]  ## 1 x 2
        bbox_scale = data_samples["meta"]["bbox_scale"]  ## 1 x 2

        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )
        keypoints.append(keypoints_i[0])  ## remove fake batch dim
        keypoint_scores.append(keypoint_scores_i[0])  ## remove fake batch dim

    return keypoints, keypoint_scores, bboxes


# -------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving per-video predictions JSON (saved by default).",
    )
    parser.add_argument(
        "--predictions-name",
        default=None,
        help="Override predictions JSON filename (used by helper for per-chunk writes).",
    )

    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    ## add pose metainfo to model
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file="configs/_base_/keypoints308.py")
        )

    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = mmdet_pipeline(detector.cfg)

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

    frames_records = []
    image_size = None
    num_keypoints_seen = None

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        try:
            keypoints, keypoint_scores, bboxes = process_one_image(
                args, image, detector, model
            )
        except Exception as e:
            print(f"[vis_pose] inference failed on {image_name}: {e}")
            continue

        if image_size is None:
            image_size = [int(image.shape[0]), int(image.shape[1])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image_rgb = visualize_keypoints(
            image=image_rgb,
            keypoints=keypoints,
            keypoints_visible=np.ones_like(keypoint_scores) > 0,
            keypoint_scores=keypoint_scores,
            radius=args.radius,
            thickness=args.thickness,
            kpt_thr=args.kpt_thr,
            skeleton=model.pose_metainfo["skeleton_links"],
            kpt_color=model.pose_metainfo["keypoint_colors"],
            link_color=model.pose_metainfo["skeleton_link_colors"],
        )
        vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(args.output, image_name)
        cv2.imwrite(save_path, vis_image)

        if not args.no_save_json:
            try:
                instances = []
                for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
                    instances.append({
                        "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                        "keypoints": np.asarray(kpts, dtype=float).tolist(),
                        "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                    })
                frames_records.append({
                    "image_name": image_name,
                    "instances": instances,
                })
            except Exception as e:
                print(f"[vis_pose] json record failed on {image_name}: {e}")

    if not args.no_save_json:
        nn = os.path.basename(os.path.normpath(args.output))
        # strip a trailing "_output" suffix so the JSON sidecar name matches the
        # video basename (e.g. ".../v3/01/<ckpt>_output/01_predictions.json").
        # `loop.sh` wraps each video output as `<video>/<ckpt>_output/`, with the
        # video number sitting one directory up.
        parent_nn = os.path.basename(os.path.dirname(os.path.normpath(args.output)))
        video_label = parent_nn if nn.endswith("_output") else nn
        json_filename = args.predictions_name or f"{video_label}_predictions.json"
        json_path = os.path.join(args.output, json_filename)
        payload = {
            "video": video_label,
            "image_size": image_size,
            "num_keypoints": num_keypoints_seen,
            "kpt_thr_used": float(args.kpt_thr),
            "frames": frames_records,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)
        print(f"[vis_pose] wrote predictions: {json_path} ({len(frames_records)} frames)")


if __name__ == "__main__":
    main()
