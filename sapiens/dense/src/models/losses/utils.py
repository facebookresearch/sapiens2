# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import functools
from typing import Callable

import torch
from torch import Tensor


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce the loss tensor based on reduction type.

    Args:
        loss: Loss tensor to reduce.
        reduction: Reduction type ('none', 'mean', or 'sum').

    Returns:
        Reduced loss tensor.
    """
    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Unknown reduction type: {reduction}")


def weight_reduce_loss(
    loss: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Apply weight and reduction to loss tensor.

    Args:
        loss: Loss tensor.
        weight: Optional element-wise weight.
        reduction: Reduction type ('none', 'mean', or 'sum').
        avg_factor: Optional averaging factor.

    Returns:
        Weighted and reduced loss tensor.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(
    loss_func: Callable[..., Tensor],
) -> Callable[..., Tensor]:
    """Decorator to add weight and reduction support to a loss function.

    Args:
        loss_func: Loss function to wrap.

    Returns:
        Wrapped loss function with weight and reduction support.
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: str = "mean",
        avg_factor: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
