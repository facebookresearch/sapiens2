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
num_iters = 4e4  ## 32 nodes, 8 gpus: 256 gpus. bs: 3, global bs: 768. num samples: 1e6. 1e6/768 = 1302. 1 epoch = 1e3 iters.
# num_iters = 1e4  ## light finetune

# ------------------------------------------------------------------------------
vis_every_iters = 100
log_every_iters = 10
save_every_iters = 1000
val_every_iters = 1000

# # debug
# vis_every_iters = 1
# log_every_iters = 1
# val_every_iters = 2
# save_every_iters = 1000

load_from = None
resume = False

# ------------------------------------------------------------------
model_name = "sapiens2_5b"
embed_dim = 2432
num_layers = 56
num_heads = 32
layer_decay_rate = 0.94

pretrained_checkpoint = f"{_CHECKPOINT_ROOT}/pretrain/sapiens2_5b_pretrain.safetensors"

##-----------------------------------------------------------------
image_size = (1024, 768)  ## height x width
patch_size = 16

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
        # mixed_precision="bf16",  # Options: ‘no’,‘fp16’,‘bf16’ or ‘fp8’.
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
        # parallelism_cfg=dict(
        #     dp_shard_size=2,  # Fully Sharded Data Parallel degree
        #     dp_replicate_size=1,  # Data Parallel degree
        #     tp_size=1,  # Tensor Parallel degree
        #     cp_size=4,  # Context Parallel degree
        # ),
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
    type="NormalVisualizer",
    vis_interval=vis_every_iters,
    vis_max_samples=4,
    vis_image_width=384,
    vis_image_height=512,
)


##-----------------------------------------------------------------
train_pipeline = [
    dict(type="PhotoMetricDistortion"),
    dict(type="RandomDownUpSampleImage", scale_range=(0.1, 0.7), prob=0.2),
    dict(
        type="NormalRandomScale",
        scale_min=0.5,
        scale_max=2.0,
        prob=0.3,
    ),
    dict(
        type="NormalRandomCropContinuous",
        ar_range=(0.5, 2.0),
        area_range=(0.4, 1.0),
        num_attempts=8,
        prob=0.3,
    ),
    dict(
        type="NormalRandomFlip",
        prob=0.3,
    ),
    dict(type="NormalResize", height=1024, width=768),
    dict(
        type="RandomGaussianBlur", prob=0.3, kernel_size=(3, 3), sigma_range=(0.1, 2.0)
    ),
    dict(type="RandomGaussianNoise", prob=0.3, var_range=(5.0, 20.0)),
    dict(type="RandomSolarize", prob=0.3, threshold=128),
    dict(type="NormalGenerateTarget"),
    dict(
        type="NormalPackInputs",
        meta_keys=(
            "img_path",
            "ori_shape",
        ),
    ),
]

val_pipeline = [
    dict(type="NormalResize", height=1024, width=768, test_mode=True),
    dict(
        type="NormalPackInputs",
        test_mode=True,
        meta_keys=(
            "img_path",
            "orig_img_height",
            "orig_img_width",
            "img_shape",
            "pad_shape",
        ),
    ),
]

test_pipeline = [
    dict(type="NormalResizePadImage", height=1024, width=768, pad_val=0),
    dict(
        type="NormalPackInputs",
        meta_keys=(
            "img_path",
            "orig_img_height",
            "orig_img_width",
            "img_shape",
        ),
    ),
]

metasim_dataset = dict(
    type="NormalMetaSimDataset",
    airstore_template="airstore://codec_avatar_sapiens_metasim_v1_no_user_data",
    json_path=f"{_DATA_ROOT}/seg/data/metasim/meta_data_v1.json",
)

render_people_dataset = dict(
    type="NormalRenderPeopleBodyDataset",  ## body only
    data_root=f"{_DATA_ROOT}/synthetic",
    seg_data_root=f"{_DATA_ROOT}/RenderPeople/part_seg",
)

multihuman_render_people_dataset = dict(
    type="NormalRenderPeopleMultihumanDataset",
    data_root=f"{_DATA_ROOT}/seg/data/render_people/synthetic_multi_human",
    normal_extension=".npz",
    seg_data_root=f"{_DATA_ROOT}/seg/data/render_people/synthetic_multi_human/part_seg",  ## supervise on face for multihuman
)

# train_datasets = 2 * [metasim_dataset] + [
#     render_people_dataset,
#     multihuman_render_people_dataset,
# ]

# train_datasets = [render_people_dataset]
# train_datasets = [multihuman_render_people_dataset]
train_datasets = [metasim_dataset]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
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
    multiprocessing_context="spawn",
    # num_workers=0, # debug
    # persistent_workers=False, # debug
    shuffle=False,
    dataset=dict(
        type="NormalRenderPeopleBodyDataset",  ## body only
        # num_samples=100,  ## debug: only use N samples for validation
        test_mode=True,
        data_root=f"{_DATA_ROOT}/seg/data/metasim/evaluation",
        pipeline=val_pipeline,
    ),
)

val_cfg = dict(
    val_interval=val_every_iters,
    evaluator=dict(
        type="NormalEvaluator",
    ),
)

data_preprocessor = dict(
    type="ImagePreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,  ## convert from bgr to rgb for pretrained models
)

##-----------------------------------------------------------------
model = dict(
    type="NormalEstimator",
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
        type="NormalHead",
        in_channels=embed_dim,
        upsample_channels=[1536, 768, 512, 256],  ## 1K resolution
        conv_out_channels=[128, 64, 32],
        conv_kernel_sizes=[3, 3, 3],
        loss_decode=[
            dict(
                type="NormalCosineSimilarityLoss",
                loss_weight=10.0,
            ),
            dict(type="L1Loss", loss_weight=1.0),
            dict(type="NormalGradL1Loss", loss_weight=10.0),
        ],
    ),
)


##-----------------------------------------------------------------
optimizer = dict(
    type="AdamW",
    # lr=5e-4,
    lr=1e-4,
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
