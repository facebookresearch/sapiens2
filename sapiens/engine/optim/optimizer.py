# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import OPTIMIZERS
from torch.optim import Adam, AdamW, SGD

OPTIMIZERS.register_module(name="AdamW")(AdamW)
OPTIMIZERS.register_module(name="Adam")(Adam)
OPTIMIZERS.register_module(name="SGD")(SGD)
