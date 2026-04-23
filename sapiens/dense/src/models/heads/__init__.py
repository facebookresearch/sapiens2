# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .albedo_head import AlbedoHead
from .normal_head import NormalHead
from .pointmap_head import PointmapHead
from .seg_head import SegHead

__all__ = ["PointmapHead", "SegHead", "NormalHead", "AlbedoHead"]
