# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS


def init_model(
    config: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = None,
    device: str = "cuda:0",
):
    assert isinstance(config, (str, Path))
    assert checkpoint is None or isinstance(checkpoint, (str, Path))

    config = Config.fromfile(config)

    ## avoid loading the pretrained backbone weights
    if "init_cfg" in config.model["backbone"]:
        config.model["backbone"].pop("init_cfg")

    model = MODELS.build(config.model)
    data_preprocessor = MODELS.build(config.data_preprocessor)

    if checkpoint is not None:
        if str(checkpoint).endswith(".safetensors"):
            state_dict = load_file(checkpoint, device="cpu")
        else:  # Handle .pth and .bin files
            checkpoint_data = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            state_dict = (
                checkpoint_data["state_dict"]
                if "state_dict" in checkpoint_data
                else checkpoint_data["model"]
            )

        incompat = model.load_state_dict(state_dict, strict=False)

        if incompat.missing_keys:
            print(f"Missing keys: {incompat.missing_keys}")

        if incompat.unexpected_keys:
            print(f"Unexpected keys: {incompat.unexpected_keys}")

        print(f"\033[96mModel loaded from {checkpoint}\033[0m")

    model.cfg = config
    model.data_preprocessor = data_preprocessor
    model.pipeline = Compose(config.test_pipeline)

    model.to(device)
    model.eval()

    return model
