# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import math
import socket
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
from sapiens.registry import MODELS
from torch import nn, Tensor


def is_list_of(seq: Any, expected_type: Union[type, tuple[type, ...]]) -> bool:
    """Check if sequence is list of expected type."""
    if not isinstance(seq, list):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def _no_grad_trunc_normal_(
    tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ----------------------------------------------------------------------------
@MODELS.register_module()
class BaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        if self.init_cfg is None:
            return

        if not isinstance(self.init_cfg, dict):
            raise TypeError(f"init_cfg must be a dict, got {type(self.init_cfg)}")

        init_type = self.init_cfg.get("type", "")

        if init_type == "Pretrained":
            checkpoint_path = self.init_cfg.get("checkpoint")
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint path must be provided for Pretrained init_cfg"
                )
            self._load_checkpoint(checkpoint_path)
        elif init_type == "":
            raise ValueError("init_cfg must specify a 'type' field")
        else:
            raise ValueError(f"Unsupported init_cfg type: {init_type}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        servername = socket.gethostname().split(".")[0]

        if rank == 0:
            from sapiens.engine.logger import Logger

            logger = Logger.get_current_instance()
            logger.info(f"Loading checkpoint from {checkpoint_path} on {servername}.")

        try:
            if checkpoint_path.endswith(".safetensors"):
                state_dict = safetensors.torch.load_file(checkpoint_path)
            else:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )

                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "teacher" in checkpoint:
                    state_dict = checkpoint["teacher"]
                    # Remove 'backbone.' prefix from state_dict keys if present
                    state_dict = {
                        key.replace("backbone.", "", 1)
                        if key.startswith("backbone.")
                        else key: value
                        for key, value in state_dict.items()
                    }
                else:
                    state_dict = checkpoint

            # Load state dict with strict=False to allow partial loading
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys and rank == 0:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys and rank == 0:
                logger.warning(
                    f"Unexpected keys when loading checkpoint: {unexpected_keys}"
                )

            if rank == 0:
                logger.info(f"Checkpoint {checkpoint_path} loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor):
        """Forward function."""
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore
