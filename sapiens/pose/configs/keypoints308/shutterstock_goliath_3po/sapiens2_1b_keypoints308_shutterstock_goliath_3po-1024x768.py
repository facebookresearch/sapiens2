# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

_CHECKPOINT_ROOT = os.path.expanduser(
    os.environ.get("SAPIENS_CHECKPOINT_ROOT", "~/sapiens2_host")
)
_DATA_ROOT = os.path.expanduser(os.environ.get("DATA_ROOT", "~/sapiens_data"))

warmup_iters = 500
# num_iters = 4e4
num_iters = 2e4

# ------------------------------------------------------------------------------
vis_every_iters = 100
log_every_iters = 10
save_every_iters = 1000
val_every_iters = 1000

# # # # debug
# vis_every_iters = 1
# log_every_iters = 1
# val_every_iters = 2
# save_every_iters = 1000

load_from = None
resume = False

# ------------------------------------------------------------------
model_name = "sapiens2_1b"
embed_dim = 1536
num_layers = 40
num_heads = 24

layer_decay_rate = 0.9
pretrained_checkpoint = f"{_CHECKPOINT_ROOT}/pretrain/sapiens2_1b_pretrain.safetensors"

##-----------------------------------------------------------------
image_size = (1024, 768)  ## height x width
patch_size = 16

sigma = 6  ## sigma is 2 for 256
scale = 4
num_keypoints = 308

# ------------------------------------------------------------------
use_fsdp = True
# use_fsdp = False

use_compile = True
# use_compile = False

## DDP config
if use_fsdp is False:
    accelerator_cfg = dict(
        type="DDP",
        log_with="tensorboard",
        # find_unused_parameters=True,
        gradient_accumulation_steps=1,  # only accumulation=1 is supported. Otherwise, the LR scheduler will be off.
        max_interval=num_iters,
        mixed_precision="bf16",  # Options: ‘no’,‘fp16’,‘bf16’ or ‘fp8’.
        step_scheduler_with_optimizer=False,  ## schedule independent of n_gpus
    )

else:
    accelerator_cfg = dict(
        type="FSDP",
        log_with="tensorboard",
        gradient_accumulation_steps=1,  # only accumulation=1 is supported. Otherwise, the LR scheduler will be off.
        max_interval=num_iters,
        mixed_precision="bf16",  # Options: ‘no’,‘fp16’,‘bf16’ or ‘fp8’.
        step_scheduler_with_optimizer=False,
        fsdp_cfg=dict(
            fsdp_version=2,  # DTensor-based engine
            state_dict_type="SHARDED_STATE_DICT",  # SHARDED_STATE_DICT | FULL_STATE_DICT
            mixed_precision=dict(
                param_dtype="bf16",
                reduce_dtype="bf16",
            ),
            cpu_ram_efficient_loading=False,
        ),
    )

if use_compile:
    accelerator_cfg["compile_cfg"] = dict(
        backend="inductor",
        mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
        fullgraph=False,
        dynamic=False,
    )

# ------------------------------------------------------------------
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
logger = dict(
    type="Logger",
    log_interval=log_every_iters,
)
checkpoint = dict(
    type="Checkpointer",
    save_interval=save_every_iters,
)

visualizer = dict(
    type="PoseVisualizer",
    vis_interval=vis_every_iters,
    vis_max_samples=4,
    vis_image_width=384,
    vis_image_height=512,
    num_keypoints=num_keypoints,
)


##-----------------------------------------------------------------
codec = dict(
    type="UDPHeatmap",
    input_size=(image_size[1], image_size[0]),  ## width x height
    heatmap_size=(int(image_size[1] / scale), int(image_size[0] / scale)),
    sigma=sigma,
)  ## sigma is 2 for 256

train_pipeline = [
    dict(type="PoseGetBBoxCenterScale"),
    dict(type="PoseRandomFlip", direction="horizontal"),  ## default prob is 0.5
    dict(type="PoseRandomHalfBody"),
    dict(type="PoseRandomBBoxTransform"),
    dict(type="PoseTopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="RandomPhotoMetricDistortion", prob=0.8),
    dict(
        type="PoseAlbumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="PoseGenerateTarget", encoder=codec),
    dict(type="PosePackInputs"),
]

val_pipeline = [
    dict(type="PoseGetBBoxCenterScale"),
    dict(type="PoseTopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PosePackInputs"),
]

test_pipeline = [
    dict(type="PoseGetBBoxCenterScale"),
    dict(type="PoseTopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PosePackInputs"),
]

##------------------------------------------------------------------------
dataset_shutterstock_train = dict(
    type="Keypoints308ShutterstockDataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_102866/itw_shutterstock_body_keypoint_344_train:2025070300.json",
)

dataset_goliath_train = dict(
    type="Keypoints308GoliathDataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/sociopticon_body_keypoint_344_train:2024093001.json",
    subsample_factor=8,
)

dataset_3po_train = dict(
    type="Keypoints308_3PODataset",
    ann_file=f"{_DATA_ROOT}/indices/3po/train.json",
    subsample_factor=2,
)

# train_datasets = [dataset_shutterstock_train]
# train_datasets = [dataset_goliath_train]
# train_datasets = [dataset_3po_train]
train_datasets = (
    [dataset_goliath_train] + 2 * [dataset_shutterstock_train] + [dataset_3po_train]
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="CombinedDataset", datasets=train_datasets, pipeline=train_pipeline
    ),
)

# ------------------------------------------------------------------------------
dataset_shutterstock_val = dict(
    type="Keypoints308ShutterstockEvalDataset",
    data_root=f"{_DATA_ROOT}/pose/data/shutterstock/test/images",
    ann_file=f"{_DATA_ROOT}/pose/data/shutterstock/test/annotations/person_keypoints_test2025_1k.json",
    test_mode=True,
    # num_samples=10,  ## debug
    pipeline=val_pipeline,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    multiprocessing_context="spawn",  ## avoids fork error with airstore
    # num_workers=0,  # debug
    # persistent_workers=False,  # debug
    shuffle=False,
    dataset=dataset_shutterstock_val,
    collate_fn=dict(type="eval_collate"),
)

val_cfg = dict(
    val_interval=val_every_iters,
    flip_test=True,  ## left right flip
    evaluator=dict(
        type="Keypoints308Evaluator",
        decoder=codec,
        ann_file=f"{_DATA_ROOT}/pose/data/shutterstock/test/annotations/person_keypoints_test2025_1k.json",
    ),
)

# dataset_goliath_val = dict(
#     type="Keypoints308GoliathEvalDataset",
#     data_root=f"{_DATA_ROOT}/pose/data/goliath/test_10000/images",
#     ann_file=f"{_DATA_ROOT}/pose/data/goliath/test_10000/annotations/person_keypoints_test2023.json",
#     test_mode=True,
#     # num_samples=10,  ## debug
#     pipeline=val_pipeline,
# )

# val_dataloader = dict(
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     multiprocessing_context="spawn",  ## avoids fork error with airstore
#     # num_workers=0,  # debug
#     # persistent_workers=False,  # debug
#     shuffle=False,
#     dataset=dataset_goliath_val,
#     collate_fn=dict(type="eval_collate"),
# )

# val_cfg = dict(
#     val_interval=val_every_iters,
#     flip_test=True,  ## left right flip
#     evaluator=dict(
#         type="Keypoints308Evaluator",
#         decoder=codec,
#         ann_file=f"{_DATA_ROOT}/pose/data/goliath/test_10000/annotations/person_keypoints_test2023.json",
#     ),
# )

data_preprocessor = dict(
    type="ImagePreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  ## convert from bgr to rgb for pretrained models
)

##-----------------------------------------------------------------
model = dict(
    type="PoseTopdownEstimator",
    backbone=dict(
        type="Sapiens2",
        arch=model_name,
        img_size=image_size,
        patch_size=patch_size,
        final_norm=True,
        use_tokenizer=False,
        with_cls_token=True,
        out_type="featmap",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_checkpoint),
    ),
    decode_head=dict(
        type="PoseHeatmapHead",
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(1536, 1024),  ## this will 2x at each step. so total is 4x
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 512, 256),
        conv_kernel_sizes=(1, 1, 1),
        loss_decode=dict(
            type="KeypointMSELoss", use_target_weight=True, loss_weight=10.0
        ),
        # loss_decode=dict(type='KeypointOHKMMSELoss', use_target_weight=True, topk=128), ## loss only for top 128 keypoints. for finetuning later.
    ),
)


##-----------------------------------------------------------------
optimizer = dict(
    type="AdamW",
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=layer_decay_rate,
    ),
    fused=True,
)

scheduler = dict(
    type="SequentialLR",
    milestones=[warmup_iters],
    schedulers=[
        dict(type="LinearLR", start_factor=1e-3, total_iters=warmup_iters),
        dict(
            type="PolynomialLR",
            total_iters=num_iters - warmup_iters,
            power=1.0,
        ),
    ],
)

clip_grad = dict(mode="norm", max_norm=4.0, norm_type=2.0)

runner_type = "PoseRunner"
