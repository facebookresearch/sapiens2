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
model_name = "sapiens2_0.4b"
embed_dim = 1024
num_layers = 24
num_heads = 16

layer_decay_rate = 0.8
pretrained_checkpoint = f"{_CHECKPOINT_ROOT}/pretrain/sapiens2_0.4b_pretrain.safetensors"

num_classes = 29  ## 29 classes
CLASS_WEIGHT = [
    0.1,
    10,
    10,
    3,
    2,
    4,
    4,
    2,
    2,
    6,
    10,
    3,
    3,
    1,
    4,
    4,
    2,
    2,
    6,
    10,
    3,
    3,
    1,
    1,
    10,
    10,
    10,
    10,
    10,
]  ## 29 classes

##-----------------------------------------------------------------
image_size = (1024, 768)  ## height x width
patch_size = 16

# ------------------------------------------------------------------
# use_fsdp = True
use_fsdp = False

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
            # state_dict_type="FULL_STATE_DICT",  # TODO: resume from this is not working
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
    type="SegVisualizer",
    vis_interval=vis_every_iters,
    vis_max_samples=4,
    vis_image_width=384,
    vis_image_height=512,
    class_palette_type="dome29",
)


##-----------------------------------------------------------------
train_pipeline = [
    dict(
        type="SegRandomBackground",
        prob=0.8,
        skip_key="is_itw",
        background_images_root=f"{_DATA_ROOT}/BG-20k/train",
    ),
    dict(
        type="SegRandomResize",
        base_height=1024,
        base_width=768,
        ratio_range=(0.4, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="SegRandomCrop",
        crop_height=1024,
        crop_width=768,
        prob=0.3,
        cat_max_ratio=0.75,
    ),
    dict(
        type="RandomGaussianBlur", prob=0.3, kernel_size=(3, 3), sigma_range=(0.1, 2.0)
    ),
    dict(type="RandomGaussianNoise", prob=0.3, var_range=(5.0, 20.0)),
    dict(
        type="SegRandomRotate", prob=0.5, degree=60, seg_pad_val=0
    ),  ## the black pixels are set as background
    dict(
        type="SegRandomHorizontalFlip",
        prob=0.5,
        swap_seg_labels=[
            (5, 14),
            (6, 15),
            (7, 16),
            (8, 17),
            (9, 18),
            (10, 19),
            (11, 20),
            (12, 21),
        ],
    ),  ## for the 29 classes,
    dict(type="PhotoMetricDistortion"),
    dict(type="SegResize", height=1024, width=768, keep_ratio=False),
    dict(type="SegPackInputs"),
]

val_pipeline = [
    dict(type="SegResize", height=1024, width=768, keep_ratio=False, test_mode=True),
    dict(type="SegPackInputs", test_mode=True),
]

test_pipeline = [
    dict(type="SegResize", height=1024, width=768, keep_ratio=False, test_mode=True),
    dict(type="SegPackInputs", test_mode=True),
]

##------------------------------------------------------------------------
dataset_dome_train = dict(
    type="SegDomeClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/sociopticon_body_segmentation_33_train:2024092600.json",
)

dataset_shutterstock_train = dict(
    type="SegShutterstockClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/itw_shutterstock_body_segmentation_51_train:2024121600.json",
)

dataset_ca3_wide_train = dict(
    type="SegDomeClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/ca3_wide_angle_body_segmentation_33_train:2024091700.json",
)

dataset_caa_train = dict(
    type="SegDomeClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/cca_segmentation_33_train:2024092400.json",
)

dataset_ca3_zoom_train = dict(
    type="SegShutterstockClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/ca3_zoom_in_body_segmentation_50_train:2024091700.json",
)

dataset_lighticon_train = dict(
    type="SegShutterstockClass29Dataset",
    ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/lighticon_lightful_body_segmentation_51_train:2025021900.json",
)

dataset_internal_train = dict(
    type="SegInternalClass29Dataset",
    # ann_file=f"{_DATA_ROOT}/annotations/stylized_sapiens/20250807/Internal_segmentation_32:2025080700.json",
    ann_file=f"{_DATA_ROOT}/annotations/internal_dataset/20251103/internal_keypoint_344_segmentation_32_train:2025091500.json",
)

train_datasets = [
    dataset_dome_train,
    dataset_ca3_wide_train,
    dataset_caa_train,
    dataset_ca3_zoom_train,
    dataset_lighticon_train,
    dataset_internal_train,
] + 2 * [dataset_shutterstock_train]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    shuffle=True,
    dataset=dict(
        type="CombinedDataset", datasets=train_datasets, pipeline=train_pipeline
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    multiprocessing_context="spawn",  ## avoids fork error with airstore
    # num_workers=0, # debug
    # persistent_workers=False, # debug
    shuffle=False,
    dataset=dict(
        type="SegShutterstockClass29Dataset",
        ann_file=f"{_DATA_ROOT}/annotations/ingestion_90942/itw_shutterstock_body_segmentation_51_test:2024121600.json",
        test_mode=True,
        pipeline=val_pipeline,
    ),
    collate_fn=dict(type="eval_collate"),
)

val_cfg = dict(
    val_interval=val_every_iters,
    evaluator=dict(type="SegEvaluator", class_names="dome29", nan_to_num=0.0),
)

data_preprocessor = dict(
    type="ImagePreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  ## convert from bgr to rgb for pretrained models
)

##-----------------------------------------------------------------
model = dict(
    type="SegEstimator",
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
        type="SegHead",
        in_channels=embed_dim,
        deconv_out_channels=(
            512,
            256,
            128,
            64,
        ),  ## this will 2x at each step. so total is 16x. 1K output.
        deconv_kernel_sizes=(4, 4, 4, 4),
        conv_out_channels=(64, 64),
        conv_kernel_sizes=(1, 1),
        num_classes=num_classes,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
                reduction="none",
                class_weight=CLASS_WEIGHT,
                ignore_index=255,
            ),
            dict(
                type="DiceLoss",
                loss_weight=1.0,
                reduction="none",
                activate=True,
                use_sigmoid=False,
                include_background=False,
                ignore_index=255,
            ),
        ],
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
    fused=True,  ## use fused AdamW
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

clip_grad = dict(mode="norm", max_norm=2.0, norm_type=2.0)
