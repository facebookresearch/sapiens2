# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sapiens.registry import SCHEDULERS
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    PolynomialLR,
    SequentialLR as _SequentialLR,
    StepLR,
)

SCHEDULERS.register_module(name="LinearLR")(LinearLR)
SCHEDULERS.register_module(name="PolynomialLR")(PolynomialLR)
SCHEDULERS.register_module(name="CosineAnnealingLR")(CosineAnnealingLR)
SCHEDULERS.register_module(name="ConstantLR")(ConstantLR)
SCHEDULERS.register_module(name="StepLR")(StepLR)
SCHEDULERS.register_module(name="MultiStepLR")(MultiStepLR)
SCHEDULERS.register_module(name="ExponentialLR")(ExponentialLR)


# ------------------------------------------------------------------------- #
@SCHEDULERS.register_module(name="SequentialLR")
class SequentialLR(_SequentialLR):
    """SequentialLR that accepts inner schedulers as config dicts.

    Example (iteration based):

    ```python
    warmup_iters = 400
    param_scheduler = dict(
        type="SequentialLR",
        milestones=[warmup_iters],
        schedulers=[
            dict(type="LinearLR",     start_factor=1e-3,
                 total_iters=warmup_iters),
            dict(type="PolynomialLR", total_iters=num_iters-warmup_iters,
                 power=1.0),
        ],
    )
    ```
    """

    def __init__(
        self,
        optimizer,
        schedulers,
        milestones,
        last_epoch: int = -1,
    ):
        built = [
            s
            if isinstance(s, _LRScheduler)
            else SCHEDULERS.build(s, optimizer=optimizer)
            for s in schedulers
        ]
        super().__init__(
            optimizer,
            schedulers=built,
            milestones=milestones,
            last_epoch=last_epoch,
        )
