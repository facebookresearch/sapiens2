# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from sapiens.registry import MODELS


@MODELS.register_module()
class BaseEvaluator:
    def __init__(self, dtype: torch.dtype = torch.float32):
        assert torch.cuda.is_available(), "CUDA is required for evaluation"
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.results = []

    def reset(self):
        self.results: List[Union[Dict[str, Any], List[Any], tuple]] = []

    def process(self, outputs, data_samples):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
